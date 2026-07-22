# src/ner/mock_data.py
"""Synthetic DocBin generator for the Track A NER plumbing test."""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LOG = logging.getLogger("mrsa_nlp.ner.mock_data")

ENTITY_LABELS = {"DISEASE", "MEDICATION", "PROCEDURE"}
SPLIT_BY_PATIENT = {
    "train": {"P001", "P002", "P003"},
    "val": {"P004"},
    "test": {"P005", "P006"},
}


@dataclass
class MockNERDataConfig:
    out_dir: Path = Path("annotations/mock")
    seed: int = 7


def _mock_note_specs() -> List[Dict]:
    return [
        {
            "patient_id": "P001",
            "note_id": "MOCK-N001",
            "text": "Patient has diabetes mellitus and takes vancomycin. A central line is present.",
            "entities": [
                ("diabetes mellitus", "DISEASE", {}),
                ("vancomycin", "MEDICATION", {}),
                ("central line", "PROCEDURE", {}),
            ],
        },
        {
            "patient_id": "P001",
            "note_id": "MOCK-N002",
            "text": "No pneumonia is seen today. Prednisone was continued after wound debridement.",
            "entities": [
                ("pneumonia", "DISEASE", {"assertion": "NEGATED"}),
                ("Prednisone", "MEDICATION", {}),
                ("wound debridement", "PROCEDURE", {}),
            ],
        },
        {
            "patient_id": "P002",
            "note_id": "MOCK-N003",
            "text": "History of MRSA bacteremia in 2020. Patient receives hemodialysis twice weekly.",
            "entities": [
                ("MRSA bacteremia", "DISEASE", {"temporality": "HISTORICAL"}),
                ("hemodialysis", "PROCEDURE", {}),
            ],
        },
        {
            "patient_id": "P002",
            "note_id": "MOCK-N004",
            "text": "Tacrolimus was started after kidney transplant. Cellulitis is improving.",
            "entities": [
                ("Tacrolimus", "MEDICATION", {}),
                ("kidney transplant", "PROCEDURE", {"temporality": "HISTORICAL"}),
                ("Cellulitis", "DISEASE", {}),
            ],
        },
        {
            "patient_id": "P003",
            "note_id": "MOCK-N005",
            "text": "Mother has lymphoma. Patient denies fever and uses methotrexate.",
            "entities": [
                ("lymphoma", "DISEASE", {"experiencer": "OTHER"}),
                ("fever", "DISEASE", {"assertion": "NEGATED"}),
                ("methotrexate", "MEDICATION", {}),
            ],
        },
        {
            "patient_id": "P003",
            "note_id": "MOCK-N006",
            "text": "Possible urinary tract infection noted. Foley catheter removed yesterday.",
            "entities": [
                ("urinary tract infection", "DISEASE", {"assertion": "UNCERTAIN"}),
                ("Foley catheter", "PROCEDURE", {}),
            ],
        },
        {
            "patient_id": "P004",
            "note_id": "MOCK-N007",
            "text": "Ciprofloxacin planned before surgery tomorrow. No abscess on exam.",
            "entities": [
                ("Ciprofloxacin", "MEDICATION", {"temporality": "PLANNED"}),
                ("surgery", "PROCEDURE", {"temporality": "PLANNED"}),
                ("abscess", "DISEASE", {"assertion": "NEGATED"}),
            ],
        },
        {
            "patient_id": "P004",
            "note_id": "MOCK-N008",
            "text": "Sepsis resolved after piperacillin-tazobactam. PICC line remains in place.",
            "entities": [
                ("Sepsis", "DISEASE", {"temporality": "HISTORICAL"}),
                ("piperacillin-tazobactam", "MEDICATION", {}),
                ("PICC line", "PROCEDURE", {}),
            ],
        },
        {
            "patient_id": "P005",
            "note_id": "MOCK-N009",
            "text": "Patient reports chronic kidney disease. Azathioprine dose unchanged.",
            "entities": [
                ("chronic kidney disease", "DISEASE", {}),
                ("Azathioprine", "MEDICATION", {}),
            ],
        },
        {
            "patient_id": "P006",
            "note_id": "MOCK-N010",
            "text": "Chest tube placed for pleural fluid. No evidence of bacteremia.",
            "entities": [
                ("Chest tube", "PROCEDURE", {}),
                ("bacteremia", "DISEASE", {"assertion": "NEGATED"}),
            ],
        },
    ]


def _sidecar_row(
    note_id: str,
    patient_id: str,
    start_char: int,
    end_char: int,
    label: str,
    text: str,
    attrs: Dict[str, str],
) -> Dict[str, object]:
    return {
        "note_id": note_id,
        "patient_id": patient_id,
        "start_char": start_char,
        "end_char": end_char,
        "label": label,
        "text": text,
        "assertion": attrs.get("assertion", "PRESENT"),
        "temporality": attrs.get("temporality", "CURRENT"),
        "experiencer": attrs.get("experiencer", "PATIENT"),
    }


def _write_sidecar(path: Path, rows: Iterable[Dict[str, object]]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _assert_patient_split(split_rows: Dict[str, List[Dict[str, object]]]) -> None:
    seen: Dict[str, str] = {}
    for split_name, rows in split_rows.items():
        for row in rows:
            patient_id = str(row["patient_id"])
            previous = seen.get(patient_id)
            if previous is not None and previous != split_name:
                raise AssertionError(
                    f"patient_id {patient_id} appears in both {previous} and {split_name}"
                )
            seen[patient_id] = split_name


def generate_mock_ner_data(cfg: MockNERDataConfig) -> Dict[str, Path]:
    """Generate synthetic note-level DocBins and sidecar CSVs."""
    try:
        import spacy
        from spacy.tokens import DocBin
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is required to generate .spacy DocBins. Install spaCy in the active environment."
        ) from exc

    rng = random.Random(cfg.seed)
    nlp = spacy.blank("en")
    specs = _mock_note_specs()
    rng.shuffle(specs)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    split_docs = {split: [] for split in SPLIT_BY_PATIENT}
    split_rows: Dict[str, List[Dict[str, object]]] = {split: [] for split in SPLIT_BY_PATIENT}
    alignment_failures = 0

    for spec in specs:
        patient_id = spec["patient_id"]
        note_id = spec["note_id"]
        split = next(
            split_name
            for split_name, patients in SPLIT_BY_PATIENT.items()
            if patient_id in patients
        )
        doc = nlp.make_doc(spec["text"])
        doc.user_data["patient_id"] = patient_id
        doc.user_data["note_id"] = note_id

        spans = []
        for surface, label, attrs in spec["entities"]:
            if label not in ENTITY_LABELS:
                raise ValueError(f"Unexpected mock entity label {label!r}")
            start_char = spec["text"].index(surface)
            end_char = start_char + len(surface)
            span = doc.char_span(start_char, end_char, label=label, alignment_mode="strict")
            if span is None:
                alignment_failures += 1
                LOG.warning(
                    "Mock span failed to align: note_id=%s patient_id=%s start=%d end=%d label=%s text=%r",
                    note_id,
                    patient_id,
                    start_char,
                    end_char,
                    label,
                    surface,
                )
                continue
            spans.append(span)
            split_rows[split].append(
                _sidecar_row(note_id, patient_id, start_char, end_char, label, surface, attrs)
            )

        doc.ents = spans
        split_docs[split].append(doc)

    _assert_patient_split(split_rows)

    paths: Dict[str, Path] = {}
    for split, docs in split_docs.items():
        docbin_path = cfg.out_dir / f"{split}.spacy"
        sidecar_path = cfg.out_dir / f"{split}_attributes.csv"
        DocBin(docs=docs, store_user_data=True).to_disk(docbin_path)
        _write_sidecar(sidecar_path, split_rows[split])
        paths[f"{split}_docbin"] = docbin_path
        paths[f"{split}_sidecar"] = sidecar_path
        LOG.info("Wrote %s: %d docs, %d sidecar rows", docbin_path, len(docs), len(split_rows[split]))

    summary = {
        "n_notes": sum(len(docs) for docs in split_docs.values()),
        "n_entities": sum(len(rows) for rows in split_rows.values()),
        "alignment_failures": alignment_failures,
        "splits": {
            split: {
                "n_notes": len(docs),
                "patients": sorted({doc.user_data["patient_id"] for doc in docs}),
                "sidecar_rows": len(split_rows[split]),
            }
            for split, docs in split_docs.items()
        },
    }
    summary_path = cfg.out_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    paths["summary"] = summary_path
    LOG.info("Mock data generation complete with %d alignment failures", alignment_failures)
    return paths
