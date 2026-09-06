"""Serializers for verified LLM pre-annotation artifacts.

The verified JSON files written by ``src.ner.preannotate`` are the source of
truth. These serializers are intentionally thin and do not call the LLM or redo
verification.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.ner.preannotate import ALLOWED_LABELS

LOG = logging.getLogger("mrsa_nlp.ner.preannotation_serializers")
_WEBANNO_NLP = None

@dataclass
class SerializerSummary:
    written: int = 0
    skipped_invalid_label: int = 0
    skipped_overlapping: int = 0
    alignment_failures: int = 0
    expanded_to_token_boundaries: int = 0
    snapped_token_boundaries: int = 0
    notes: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "written": self.written,
            "skipped_invalid_label": self.skipped_invalid_label,
            "skipped_overlapping": self.skipped_overlapping,
            "expanded_to_token_boundaries": self.expanded_to_token_boundaries,
            "alignment_failures": self.alignment_failures,
            "snapped_token_boundaries": self.snapped_token_boundaries,
            "notes": self.notes,
        }

    def format_block(self, title: str = "Serializer stats") -> str:
        rows = [title]
        for key, value in self.as_dict().items():
            rows.append(f"  {key}: {value}")
        return "\n".join(rows)


def load_verified_artifacts(verified_dir: Path) -> List[Dict[str, Any]]:
    """Load one-note verified JSON artifacts."""
    artifacts: List[Dict[str, Any]] = []
    for path in sorted(verified_dir.glob("*.json")):
        artifacts.append(json.loads(path.read_text()))
    return artifacts


def write_webanno_tsv3(verified_dir: Path, out_dir: Path) -> SerializerSummary:
    """Write one WebAnno TSV 3.3 file per verified note.

    Format choice:
    - Uses the built-in INCEpTION/DKPro NamedEntity layer
      ``de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|value``.
    - Annotation is spans and labels only. No assertion, temporality, or
      experiencer attributes are emitted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = SerializerSummary()
    for artifact in load_verified_artifacts(verified_dir):
        summary.notes += 1
        note_id = str(artifact["note_id"])
        text = str(artifact["text"])
        spans = [
            span
            for span in artifact.get("spans", [])
            if span.get("label") in ALLOWED_LABELS
        ]
        summary.skipped_invalid_label += len(artifact.get("spans", [])) - len(spans)
        tsv_text = _artifact_to_webanno_tsv(text, spans, note_id=note_id, summary=summary)
        out_path = out_dir / f"{_safe_filename(note_id)}.tsv"
        out_path.write_text(tsv_text)
        summary.written += 1

    (out_dir / "serializer_summary.json").write_text(json.dumps(summary.as_dict(), indent=2) + "\n")
    LOG.info("Wrote WebAnno TSV 3.3 pre-annotations to %s\n%s", out_dir, summary.format_block("WebAnno TSV stats"))
    return summary


def write_contract_docbin(verified_dir: Path, out_dir: Path, split_name: str = "preannotated") -> SerializerSummary:
    """Write CONTRACT.md-compatible DocBin and an empty attribute sidecar.

    spaCy ``Doc.ents`` cannot contain overlapping spans. Spans already flagged
    ``overlaps_previous`` are skipped here and retained in the primary verified
    JSON for human resolution.

    The sidecar is kept with required coordinate columns and empty reserved
    attribute columns because the current trainer validates that a sidecar
    exists. No LLM attribute mapping is performed.
    """
    try:
        import spacy
        from spacy.tokens import DocBin
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is required to serialize CONTRACT.md DocBin artifacts."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    nlp = spacy.blank("en")
    docbin = DocBin(store_user_data=True)
    sidecar_rows: List[Dict[str, Any]] = []
    alignment_failures: List[Dict[str, Any]] = []
    summary = SerializerSummary()

    for artifact in load_verified_artifacts(verified_dir):
        summary.notes += 1
        note_id = str(artifact["note_id"])
        patient_id = str(artifact["patient_id"])
        text = str(artifact["text"])
        doc = nlp.make_doc(text)
        doc.user_data["note_id"] = note_id
        doc.user_data["patient_id"] = patient_id
        ents = []

        for span_data in artifact.get("spans", []):
            label = span_data.get("label")
            if label not in ALLOWED_LABELS:
                summary.skipped_invalid_label += 1
                continue
            if span_data.get("overlaps_previous"):
                summary.skipped_overlapping += 1
                continue

            start_char = int(span_data["start_char"])
            end_char = int(span_data["end_char"])
            span = doc.char_span(
                start_char,
                end_char,
                label=str(label),
                alignment_mode="strict",
            )
            if span is None:
                # Strict alignment rejects spans abutting punctuation
                # ("CAD-", "COVID-19", "AMS/"). Expanding to token
                # boundaries recovers these, but only when the added
                # characters are non-alphanumeric -- otherwise expansion
                # would widen the span to a different word.
                candidate = doc.char_span(
                    start_char,
                    end_char,
                    label=str(label),
                    alignment_mode="expand",
                )
                if candidate is not None:
                    added = candidate.text.replace(
                        doc.text[start_char:end_char], "", 1
                    )
                    if not any(ch.isalnum() for ch in added):
                        span = candidate
                        summary.expanded_to_token_boundaries += 1
            if span is None:
                summary.alignment_failures += 1
                alignment_failures.append(
                    {
                        "note_id": note_id,
                        "patient_id": patient_id,
                        "start_char": start_char,
                        "end_char": end_char,
                        "label": label,
                        "text": span_data.get("text"),
                    }
                )
                continue

            ents.append(span)
            sidecar_rows.append(_contract_sidecar_row(note_id, patient_id, span_data))

        doc.ents = ents
        docbin.add(doc)
        summary.written += len(ents)

    docbin_path = out_dir / f"{split_name}.spacy"
    sidecar_path = out_dir / f"{split_name}_attributes.csv"
    failures_path = out_dir / "alignment_failures.jsonl"
    docbin.to_disk(docbin_path)
    _write_sidecar(sidecar_path, sidecar_rows)
    failures_path.write_text(
        "".join(json.dumps(item) + "\n" for item in alignment_failures)
    )
    (out_dir / "serializer_summary.json").write_text(json.dumps(summary.as_dict(), indent=2) + "\n")
    LOG.info("Wrote CONTRACT.md DocBin to %s and sidecar to %s", docbin_path, sidecar_path)
    return summary


def _contract_sidecar_row(note_id: str, patient_id: str, span: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "note_id": note_id,
        "patient_id": patient_id,
        "start_char": int(span["start_char"]),
        "end_char": int(span["end_char"]),
        "label": span["label"],
        "text": span["text"],
        "assertion": "",
        "temporality": "",
        "experiencer": "",
    }


def _write_sidecar(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fieldnames = [
        "note_id",
        "patient_id",
        "start_char",
        "end_char",
        "label",
        "text",
        "assertion",
        "temporality",
        "experiencer",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _artifact_to_webanno_tsv(
    text: str,
    spans: Sequence[Dict[str, Any]],
    note_id: str = "",
    summary: Optional[SerializerSummary] = None,
) -> str:
    doc = _get_webanno_nlp()(text)
    tokens = _spacy_sentence_tokens(doc)
    annotations_by_token: Dict[int, List[str]] = {idx: [] for idx in range(len(tokens))}

    for ann_id, span in enumerate(spans, start=1):
        covered, snapped = _tokens_covered_by_span(tokens, int(span["start_char"]), int(span["end_char"]))
        if not covered:
            continue
        if snapped:
            if summary is not None:
                summary.snapped_token_boundaries += 1
            LOG.warning(
                "Snapped WebAnno TSV span to token boundaries: note_id=%s text=%r start=%s end=%s",
                note_id,
                span.get("text"),
                span.get("start_char"),
                span.get("end_char"),
            )
        label = _escape_tsv_value(str(span["label"]))
        value = f"{label}[{ann_id}]" if len(covered) > 1 else label
        for idx in covered:
            annotations_by_token[idx].append(value)

    lines = [
        "#FORMAT=WebAnno TSV 3.3",
        "#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|value",
        "",
        "",
    ]

    for sentence_no, sentence_text, sentence_tokens in _sentence_token_groups(text, tokens):
        lines.append(f"#Text={_escape_text_header(sentence_text)}")
        for token_idx, token_no, start, end, token_text in sentence_tokens:
            values = annotations_by_token[token_idx]
            ann = "|".join(values) if values else "_"
            # `start`/`end` are Python character indices from spaCy tokens.
            # WebAnno TSV offsets are UTF-16 code-unit offsets, so convert once
            # here at the final serialization boundary.
            lines.append(
                "\t".join(
                    [
                        f"{sentence_no}-{token_no}",
                        f"{_utf16_offset(text, start)}-{_utf16_offset(text, end)}",
                        _escape_tsv_value(token_text),
                        ann,
                    ]
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _get_webanno_nlp():
    global _WEBANNO_NLP
    if _WEBANNO_NLP is None:
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("spaCy is required to serialize WebAnno TSV 3.3 artifacts.") from exc
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _WEBANNO_NLP = nlp
    return _WEBANNO_NLP


def _spacy_sentence_tokens(doc) -> List[Tuple[int, int, int, int, str]]:
    tokens: List[Tuple[int, int, int, int, str]] = []
    for sentence_no, sent in enumerate(doc.sents, start=1):
        token_no = 0
        for token in sent:
            if token.is_space:
                continue
            token_no += 1
            start = int(token.idx)
            end = start + len(token.text)
            tokens.append((sentence_no, token_no, start, end, token.text))
    return tokens


def _tokens_covered_by_span(
    tokens: Sequence[Tuple[int, int, int, int, str]],
    start_char: int,
    end_char: int,
) -> Tuple[List[int], bool]:
    exact = [
        idx
        for idx, (_sentence_no, _token_no, start, end, _token_text) in enumerate(tokens)
        if start >= start_char and end <= end_char
    ]
    start_aligned = any(start == start_char for _s, _t, start, _e, _text in tokens)
    end_aligned = any(end == end_char for _s, _t, _start, end, _text in tokens)
    if exact and start_aligned and end_aligned:
        return exact, False

    snapped = [
        idx
        for idx, (_sentence_no, _token_no, start, end, _token_text) in enumerate(tokens)
        if start < end_char and end > start_char
    ]
    return snapped, bool(snapped)


def _sentence_token_groups(
    text: str,
    tokens: Sequence[Tuple[int, int, int, int, str]],
) -> List[Tuple[int, str, List[Tuple[int, int, int, int, str]]]]:
    groups: List[Tuple[int, str, List[Tuple[int, int, int, int, str]]]] = []
    by_sentence: Dict[int, List[Tuple[int, int, int, int, str]]] = {}
    for token_idx, (sentence_no, token_no, start, end, token_text) in enumerate(tokens):
        by_sentence.setdefault(sentence_no, []).append((token_idx, token_no, start, end, token_text))
    for sentence_no in sorted(by_sentence):
        sentence_tokens = by_sentence[sentence_no]
        sentence_text = text[sentence_tokens[0][2] : sentence_tokens[-1][3]]
        groups.append((sentence_no, sentence_text, sentence_tokens))
    return groups


def _utf16_offset(text: str, char_index: int) -> int:
    return len(text[:char_index].encode("utf-16-le")) // 2


def _escape_text_header(value: str) -> str:
    return value.replace("\r", r"\r")


def _escape_tsv_value(value: str) -> str:
    value = value.replace("\\", r"\\")
    value = value.replace("\t", r"\t").replace("\n", r"\n")
    for char in [",", "[", "]", "|", "_", ";", "*"]:
        value = value.replace(char, "\\" + char)
    return value.replace("->", r"\->")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "note"
