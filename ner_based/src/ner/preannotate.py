"""LLM pre-annotation for clinical NER notes.

This stage asks a local Ollama model for candidate entity strings, then treats
the model output only as proposals. The source text is authoritative: every
accepted span is located by exact or normalized text matching before offsets
are written.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOG = logging.getLogger("mrsa_nlp.ner.preannotate")

ALLOWED_LABELS = {"DISEASE", "MEDICATION", "PROCEDURE"}

PREANNOTATION_SYSTEM_PROMPT = """You propose clinical named entity mentions for human review.

Labels:
- DISEASE: diagnoses, clinical conditions, infections, symptoms, and chronic illnesses. Examples: MRSA, pneumonia, cellulitis, sepsis, diabetes mellitus, CHF, RA, COPD, fever.
- MEDICATION: drugs, drug classes, antibiotics, immunosuppressants, and allergens when the allergen is a medication. Examples: vancomycin, prednisone, methotrexate, penicillin, antibiotics.
- PROCEDURE: clinical procedures, surgical interventions, invasive devices, indwelling devices, and lines. Indwelling devices and lines ARE procedures. Examples: central line, PICC, tunneled cath, Foley, urinary catheter, hemodialysis/HD, intubation, tracheostomy, CABG, wound debridement.

Return JSON only, with this shape:
{
  "entities": [
    {
      "text": "verbatim text copied exactly from the note",
      "label": "DISEASE|MEDICATION|PROCEDURE"
    }
  ]
}

Rules:
- The "text" value must be copied verbatim from the note. Do not paraphrase.
- Include only these two keys per entity: text and label. Do not add offsets or attributes.
- Do NOT omit negated, uncertain, historical, or family-member mentions. Tag the entity regardless of context.
- ALLERGY: "Penicillin allergy" -> tag "Penicillin" as MEDICATION.
- Do NOT tag anatomy or body parts (e.g. "L leg", "left arm"). Not entities.

Span boundaries (tag the shortest span that carries the clinical meaning):
- MEDICATION: drug name only. Exclude parenthetical brand names, dose, strength,
  form, route, and frequency.
  "omeprazole (PRILOSEC) 20 mg capsule Take 20 mg by mouth daily" -> "omeprazole"
  "fluticasone-salmeterol (ADVAIR) 250-50 mcg" -> "fluticasone-salmeterol"
- DISEASE/PROCEDURE: tag the full clinical term, but exclude trailing anatomical
  qualifiers that are separate body-part mentions.
  "Macular degeneration of right eye" -> "Macular degeneration"
  "Lumpectomy Right" -> "Lumpectomy"
- Where an abbreviation and its expansion both appear, tag each separately.
  Do not merge them into one span.
- Exclude leading bullets, numbering, and section labels from the span.

Few-shot example:
Note: No pneumonia today. Mother had lymphoma. Penicillin allergy listed. If fever develops, start vancomycin.
JSON:
{
  "entities": [
    {"text": "pneumonia", "label": "DISEASE"},
    {"text": "lymphoma", "label": "DISEASE"},
    {"text": "Penicillin", "label": "MEDICATION"},
    {"text": "fever", "label": "DISEASE"},
    {"text": "vancomycin", "label": "MEDICATION"}
  ]
}
"""


@dataclass
class Note:
    note_id: str
    patient_id: str
    text: str


@dataclass
class VerificationStats:
    proposed: int = 0
    verified: int = 0
    dropped_not_in_text: int = 0
    dropped_empty: int = 0
    ambiguous: int = 0
    overlapping: int = 0
    skipped_existing: int = 0
    parse_failed: int = 0
    request_failed: int = 0

    def add(self, other: "VerificationStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> Dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def counters_reconcile(self) -> bool:
        # `proposed` counts model-returned entity records. `ambiguous` counts
        # extra accepted occurrences beyond the first for repeated text, so:
        # proposed + ambiguous == verified + all dropped proposal counts.
        return self.proposed + self.ambiguous == (
            self.verified + self.dropped_not_in_text + self.dropped_empty
        )

    def format_block(self, title: str = "LLM pre-annotation stats") -> str:
        rows = [title]
        for key, value in self.as_dict().items():
            rows.append(f"  {key}: {value}")
        return "\n".join(rows)


@dataclass
class VerificationResult:
    note_id: str
    patient_id: str
    text: str
    spans: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    stats: VerificationStats

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "note_id": self.note_id,
            "patient_id": self.patient_id,
            "text": self.text,
            "spans": self.spans,
            "rejected": self.rejected,
            "stats": self.stats.as_dict(),
        }


@dataclass
class OllamaConfig:
    host: str
    auth_user: str
    auth_token: str
    model: str
    max_retries: int = 3

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> "OllamaConfig":
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "python-dotenv is required for real Ollama runs. Install it in the active environment."
            ) from exc

        load_dotenv()
        host = os.getenv("OLLAMA_HOST", "").rstrip("/")
        auth_user = os.getenv("OLLAMA_AUTH_USER", "")
        auth_token = os.getenv("OLLAMA_AUTH_TOKEN", "")
        resolved_model = model or os.getenv("OLLAMA_MODEL", "")
        max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

        missing = [
            name
            for name, value in {
                "OLLAMA_HOST": host,
                "OLLAMA_AUTH_USER": auth_user,
                "OLLAMA_AUTH_TOKEN": auth_token,
                "OLLAMA_MODEL or --model": resolved_model,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Ollama configuration: {', '.join(missing)}")

        return cls(
            host=host,
            auth_user=auth_user,
            auth_token=auth_token,
            model=resolved_model,
            max_retries=max_retries,
        )


class OllamaClient:
    """Small Ollama HTTP client using the Minerva bearer-token pattern."""

    def __init__(self, cfg: OllamaConfig, logger: logging.Logger = LOG) -> None:
        self.cfg = cfg
        self.log = logger

    def propose_entities(self, note_text: str) -> Dict[str, Any]:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": PREANNOTATION_SYSTEM_PROMPT},
                {"role": "user", "content": note_text},
            ],
            "format": "json",
            "options": {"temperature": 0.0, "num_ctx": 8192},
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.cfg.auth_user}:{self.cfg.auth_token}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            f"{self.cfg.host}/api/chat",
            data=data,
            headers=headers,
            method="POST",
        )

        last_error: Optional[str] = None
        for attempt in range(self.cfg.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    response_json = json.loads(response.read().decode("utf-8"))
                raw = (
                    response_json.get("message", {}).get("content")
                    or response_json.get("response")
                    or ""
                )
                return parse_model_json(str(raw), logger=self.log)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                self.log.warning(
                    "Ollama request failed on attempt %d/%d: %s",
                    attempt + 1,
                    self.cfg.max_retries,
                    last_error,
                )
                if attempt + 1 < self.cfg.max_retries:
                    time.sleep(2**attempt)

        self.log.error("Ollama request failed after retries: %s", last_error)
        return {"request_failed": True, "error": last_error or "unknown"}


def parse_model_json(raw: str, logger: logging.Logger = LOG) -> Dict[str, Any]:
    """Parse Ollama text into JSON without relying on schema enforcement."""
    stripped = _strip_markdown_fences(raw).strip()
    for candidate in (stripped, _json_object_substring(stripped)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return {"entities": parsed}
            if isinstance(parsed, dict):
                return parsed
            break
        except json.JSONDecodeError:
            continue

    logger.error("Could not parse model JSON response: %r", raw[:300])
    return {"parse_failed": True, "raw_response": raw[:300]}


def verify_model_response(note: Note, response: Mapping[str, Any]) -> VerificationResult:
    """Verify proposed entity strings against source text and assign offsets."""
    proposed_entities = _extract_entities(response)
    stats = VerificationStats(proposed=len(proposed_entities))
    if response.get("parse_failed"):
        stats.parse_failed = 1
    if response.get("request_failed"):
        stats.request_failed = 1

    spans: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for entity_index, entity in enumerate(proposed_entities):
        proposed_text = str(entity.get("text", ""))
        if not proposed_text.strip():
            stats.dropped_empty += 1
            rejected.append(
                {
                    "entity_index": entity_index,
                    "proposed": dict(entity),
                    "reason": "empty_text",
                }
            )
            continue

        occurrences = _find_exact_occurrences(note.text, proposed_text)
        if not occurrences:
            occurrences = _find_normalized_occurrences(note.text, proposed_text)

        if not occurrences:
            stats.dropped_not_in_text += 1
            rejected.append(
                {
                    "entity_index": entity_index,
                    "proposed": dict(entity),
                    "reason": "not_in_text",
                }
            )
            continue

        ambiguous = len(occurrences) > 1
        if ambiguous:
            stats.ambiguous += len(occurrences) - 1

        for occurrence_index, (start_char, end_char) in enumerate(occurrences):
            if start_char == end_char or not note.text[start_char:end_char].strip():
                stats.dropped_empty += 1
                rejected.append(
                    {
                        "entity_index": entity_index,
                        "occurrence_index": occurrence_index,
                        "proposed": dict(entity),
                        "reason": "empty_text",
                    }
                )
                continue

            span = {
                "text": note.text[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
                "label": str(entity.get("label", "")).upper(),
                "source": "llm_preannotation",
                "ambiguous_occurrence": ambiguous,
                "occurrence_index": occurrence_index if ambiguous else None,
                "overlaps_previous": False,
            }
            spans.append(span)

    _flag_overlaps(spans)
    stats.verified = len(spans)
    stats.overlapping = sum(1 for span in spans if span["overlaps_previous"])

    return VerificationResult(
        note_id=note.note_id,
        patient_id=note.patient_id,
        text=note.text,
        spans=spans,
        rejected=rejected,
        stats=stats,
    )


def run_preannotation(
    notes: Sequence[Note],
    out_dir: Path,
    client: Optional[OllamaClient] = None,
    canned_responses: Optional[Mapping[str, Mapping[str, Any]]] = None,
    overwrite: bool = False,
    logger: logging.Logger = LOG,
) -> VerificationStats:
    """Run pre-annotation and write one verified JSON artifact per note."""
    if client is None and canned_responses is None:
        raise ValueError("Either client or canned_responses must be provided")

    verified_dir = out_dir / "verified"
    verified_dir.mkdir(parents=True, exist_ok=True)

    totals = VerificationStats()
    for note in notes:
        out_path = verified_dir / f"{_safe_filename(note.note_id)}.json"
        if out_path.exists() and not overwrite:
            logger.info("Skipping existing pre-annotation artifact: %s", out_path)
            totals.skipped_existing += 1
            continue

        if canned_responses is not None:
            response = canned_responses.get(note.note_id, {"entities": []})
        else:
            assert client is not None
            response = client.propose_entities(note.text)

        result = verify_model_response(note, response)
        out_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n")
        totals.add(result.stats)
        logger.info("%s\n%s", note.note_id, result.stats.format_block("Note stats"))

    (out_dir / "run_summary.json").write_text(json.dumps(totals.as_dict(), indent=2) + "\n")
    logger.info("\n%s", totals.format_block("Run total"))
    return totals


def load_notes(path: Path) -> List[Note]:
    """Load synthetic or preprocessed notes from JSON/JSONL/TXT/parquet inputs."""
    if path.is_dir():
        notes: List[Note] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".json", ".jsonl", ".txt", ".parquet"}:
                notes.extend(load_notes(child))
        return notes

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            _note_from_mapping(json.loads(line), fallback_id=f"{path.stem}-{idx}")
            for idx, line in enumerate(path.read_text().splitlines())
            if line.strip()
        ]
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [_note_from_mapping(item, fallback_id=f"{path.stem}-{idx}") for idx, item in enumerate(data)]
        return [_note_from_mapping(data, fallback_id=path.stem)]
    if suffix == ".txt":
        return [Note(note_id=path.stem, patient_id="UNKNOWN", text=path.read_text())]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required to read parquet note inputs") from exc
        frame = pd.read_parquet(path)
        return [
            _note_from_mapping(row.to_dict(), fallback_id=f"{path.stem}-{idx}")
            for idx, row in frame.iterrows()
        ]

    raise ValueError(f"Unsupported note input format: {path}")


def synthetic_fixture_notes() -> List[Note]:
    """Synthetic notes used for offline verifier and CLI plumbing checks."""
    return [
        Note(
            note_id="SYN-N001",
            patient_id="SYN-P001",
            text=(
                "No pneumonia today. Mother had lymphoma. Patient reports fever, "
                "then fever resolved. Chronic kidney disease noted. Possible cellulitis. "
                "Penicillin allergy listed."
            ),
        ),
        Note(
            note_id="SYN-N002",
            patient_id="SYN-P002",
            text=(
                "Patient has a PICC   line and diabetes mellitus. Left arm is sore. "
                "Plan for intubation if respiratory failure develops."
            ),
        ),
    ]


def synthetic_canned_responses() -> Dict[str, Dict[str, Any]]:
    """Canned Ollama-like responses covering required verifier edge cases."""
    return {
        "SYN-N001": {
            "entities": [
                {
                    "text": "pneumonia",
                    "label": "DISEASE",
                },
                {
                    "text": "lymphoma",
                    "label": "DISEASE",
                },
                {
                    "text": "fever",
                    "label": "DISEASE",
                },
                {
                    "text": "Chronic kidney disease",
                    "label": "DISEASE",
                },
                {
                    "text": "kidney disease",
                    "label": "DISEASE",
                },
                {
                    "text": "cellulitis",
                    "label": "DISEASE",
                },
                {
                    "text": "Penicillin",
                    "label": "MEDICATION",
                },
                {
                    "text": "sepsis",
                    "label": "DISEASE",
                },
            ]
        },
        "SYN-N002": {
            "entities": [
                {
                    "text": "PICC line",
                    "label": "PROCEDURE",
                },
                {
                    "text": "diabetes mellitus",
                    "label": "DISEASE",
                },
                {
                    "text": "intubation",
                    "label": "PROCEDURE",
                },
                {
                    "text": "respiratory failure",
                    "label": "DISEASE",
                },
            ]
        },
    }


def _extract_entities(response: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    entities = response.get("entities", [])
    if not isinstance(entities, list):
        return []
    return [entity for entity in entities if isinstance(entity, Mapping)]


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text


def _json_object_substring(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return raw[start : end + 1]


def _find_exact_occurrences(text: str, needle: str) -> List[Tuple[int, int]]:
    occurrences: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            return occurrences
        occurrences.append((idx, idx + len(needle)))
        start = idx + 1


def _find_normalized_occurrences(text: str, needle: str) -> List[Tuple[int, int]]:
    normalized_text, start_map, end_map = _collapse_whitespace_with_offsets(text)
    normalized_needle = " ".join(needle.split())
    if not normalized_needle:
        return []
    matches = _find_exact_occurrences(normalized_text, normalized_needle)
    return [(start_map[start], end_map[end - 1]) for start, end in matches]


def _collapse_whitespace_with_offsets(text: str) -> Tuple[str, List[int], List[int]]:
    chars: List[str] = []
    start_map: List[int] = []
    end_map: List[int] = []
    idx = 0
    while idx < len(text):
        if text[idx].isspace():
            start = idx
            while idx < len(text) and text[idx].isspace():
                idx += 1
            chars.append(" ")
            start_map.append(start)
            end_map.append(idx)
        else:
            chars.append(text[idx])
            start_map.append(idx)
            end_map.append(idx + 1)
            idx += 1
    return "".join(chars), start_map, end_map


def _flag_overlaps(spans: List[Dict[str, Any]]) -> None:
    previous_end = -1
    for span in sorted(spans, key=lambda item: (item["start_char"], item["end_char"])):
        if span["start_char"] < previous_end:
            span["overlaps_previous"] = True
        previous_end = max(previous_end, span["end_char"])


def _note_from_mapping(data: Mapping[str, Any], fallback_id: str) -> Note:
    note_id = _first_present(data, ("note_id", "NOTE_ID", "noteid", "id"), fallback_id)
    patient_id = _first_present(
        data,
        ("patient_id", "PATIENT_ID", "PERSON_ID", "person_id", "MRN", "mrn"),
        "UNKNOWN",
    )
    text = _first_present(
        data,
        ("text", "TEXT", "note_text", "NOTE_TEXT", "clean_text", "CLEAN_TEXT"),
        "",
    )
    return Note(note_id=str(note_id), patient_id=str(patient_id), text=str(text))


def _first_present(data: Mapping[str, Any], names: Iterable[str], default: Any) -> Any:
    for name in names:
        value = data.get(name)
        if value is not None:
            return value
    return default


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "note"
