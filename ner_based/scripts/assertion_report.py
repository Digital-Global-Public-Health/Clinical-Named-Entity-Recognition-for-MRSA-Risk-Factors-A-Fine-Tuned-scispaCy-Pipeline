#!/usr/bin/env python3
"""Run AIR.MS NER + ConText over a DocBin and write a PHI-bearing review CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ner.assertion import ALLOWED_LABELS, annotate_assertions, build_assertion_pipeline


DEFAULT_DOCBIN = PROJECT_ROOT / "annotations/gold_export/gold.spacy"
DEFAULT_MODEL = PROJECT_ROOT / "models/ner_full/model-best"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/assertion_review/assertion_report.csv"
ASSERTION_FLAGS = (
    "is_negated",
    "is_historical",
    "is_hypothetical",
    "is_uncertain",
    "is_family",
)
CSV_FIELDS = (
    "note_id",
    "text",
    "label",
    "start",
    "end",
    *ASSERTION_FLAGS,
    "left_context",
    "right_context",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docbin",
        type=Path,
        default=DEFAULT_DOCBIN,
        help=f"Input DocBin (default: {DEFAULT_DOCBIN})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Fine-tuned spaCy model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Review CSV under outputs/ or annotations/ (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=60,
        help="Maximum characters of note context on each side (default: 60)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="spaCy inference batch size (default: 32)",
    )
    return parser.parse_args(argv)


def load_docbin_notes(docbin_path: Path, vocab: object) -> List[Dict[str, str]]:
    """Read note IDs and text from a contract-compatible DocBin."""
    if not docbin_path.is_file():
        raise FileNotFoundError(f"DocBin does not exist: {docbin_path}")

    from spacy.tokens import DocBin

    notes: List[Dict[str, str]] = []
    seen_note_ids = set()
    docs = DocBin().from_disk(docbin_path).get_docs(vocab)
    for doc_index, doc in enumerate(docs):
        note_id = doc.user_data.get("note_id")
        if note_id is None or str(note_id).strip() == "":
            raise ValueError(
                f"Doc {doc_index} in {docbin_path} has no doc.user_data['note_id']"
            )
        note_id = str(note_id)
        if note_id in seen_note_ids:
            raise ValueError(f"Duplicate note_id in {docbin_path}: {note_id}")
        seen_note_ids.add(note_id)
        notes.append({"note_id": note_id, "text": doc.text})
    return notes


def write_report(
    *,
    nlp: object,
    notes: Sequence[Mapping[str, str]],
    output_path: Path,
    context_chars: int,
    batch_size: int,
) -> tuple[int, Counter, Dict[str, Counter]]:
    """Write entity decisions and return totals used by the text summary."""
    if context_chars < 0:
        raise ValueError("context_chars must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    _require_phi_safe_output(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    note_texts = {note["note_id"]: note["text"] for note in notes}
    overall = Counter()
    by_label: Dict[str, Counter] = defaultdict(Counter)
    entity_count = 0

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in annotate_assertions(nlp, notes, batch_size=batch_size):
            note_text = note_texts[record["note_id"]]
            start = int(record["start"])
            end = int(record["end"])
            row = dict(record)
            row["left_context"] = note_text[max(0, start - context_chars) : start]
            row["right_context"] = note_text[end : end + context_chars]
            writer.writerow(row)

            entity_count += 1
            label = str(record["label"])
            overall["total"] += 1
            by_label[label]["total"] += 1
            for flag in ASSERTION_FLAGS:
                if record[flag]:
                    overall[flag] += 1
                    by_label[label][flag] += 1

    return entity_count, overall, by_label


def print_summary(overall: Counter, by_label: Mapping[str, Counter]) -> None:
    """Print counts and within-group percentages for every assertion flag."""
    print("\nAssertion summary (flags are non-exclusive)")
    _print_summary_group("OVERALL", overall)
    for label in sorted(ALLOWED_LABELS):
        _print_summary_group(label, by_label.get(label, Counter()))


def _print_summary_group(name: str, counts: Counter) -> None:
    total = counts["total"]
    print(f"\n{name} (n={total})")
    print(f"{'class':<20} {'count':>10} {'percent':>10}")
    for flag in ASSERTION_FLAGS:
        count = counts[flag]
        percentage = 100.0 * count / total if total else 0.0
        print(f"{flag:<20} {count:>10d} {percentage:>9.1f}%")


def _require_phi_safe_output(path: Path) -> None:
    """Refuse PHI-bearing CSV output outside already-gitignored directories."""
    resolved = path.resolve()
    safe_roots = (
        (PROJECT_ROOT / "outputs").resolve(),
        (PROJECT_ROOT / "annotations").resolve(),
    )
    if not any(resolved == root or root in resolved.parents for root in safe_roots):
        raise ValueError(
            f"PHI-bearing output must be below {safe_roots[0]} or {safe_roots[1]}: {path}"
        )
    if resolved.suffix.lower() != ".csv":
        raise ValueError(f"output must be a .csv file: {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        nlp = build_assertion_pipeline(args.model_path)
        notes = load_docbin_notes(args.docbin, nlp.vocab)
        entity_count, overall, by_label = write_report(
            nlp=nlp,
            notes=notes,
            output_path=args.output,
            context_chars=args.context_chars,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Wrote {entity_count} entity rows from {len(notes)} notes to {args.output}")
    print_summary(overall, by_label)


if __name__ == "__main__":
    main()
