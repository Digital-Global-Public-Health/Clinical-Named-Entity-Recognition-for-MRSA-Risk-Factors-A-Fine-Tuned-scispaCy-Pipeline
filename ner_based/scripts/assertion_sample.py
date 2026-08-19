#!/usr/bin/env python3
"""Draw a note-stratified manual-review sample from an assertion report CSV."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs/assertion_review/assertion_report.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/assertion_review/assertion_sample.csv"
ASSERTION_FLAGS = (
    "is_negated",
    "is_historical",
    "is_hypothetical",
    "is_uncertain",
    "is_family",
)
TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Assertion report CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Sample CSV under outputs/ or annotations/ (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-n",
        "--per-class",
        type=int,
        default=50,
        help="Maximum predicted-positive entities per assertion class (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible sampling (default: 7)",
    )
    return parser.parse_args(argv)


def read_report(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Assertion report does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        required = {"note_id", *ASSERTION_FLAGS}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"Assertion report is missing columns: {', '.join(missing)}")
        return list(reader), fieldnames


def draw_balanced_sample(
    rows: Iterable[Mapping[str, str]],
    *,
    per_class: int,
    seed: int,
) -> tuple[List[Dict[str, str]], Dict[str, int]]:
    """Sample predicted positives per flag, spreading picks across notes first.

    Flags are non-exclusive. An entity carrying multiple flags is independently
    eligible for each corresponding review class and may therefore occur more
    than once with different ``review_class`` values.
    """
    if per_class < 1:
        raise ValueError("per_class must be at least 1")

    all_rows = list(rows)
    rng = random.Random(seed)
    sampled: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}

    for flag in ASSERTION_FLAGS:
        positives = [row for row in all_rows if _is_true(row.get(flag, ""))]
        selected = _sample_across_notes(positives, per_class, rng)
        counts[flag] = len(selected)
        for row in selected:
            review_row = dict(row)
            review_row["review_class"] = flag
            review_row["correct"] = ""
            sampled.append(review_row)

    rng.shuffle(sampled)
    return sampled, counts


def _sample_across_notes(
    rows: Iterable[Mapping[str, str]],
    limit: int,
    rng: random.Random,
) -> List[Mapping[str, str]]:
    """Round-robin shuffled note pools before taking repeat rows from a note."""
    by_note: MutableMapping[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_note[str(row["note_id"])].append(row)
    for note_rows in by_note.values():
        rng.shuffle(note_rows)

    active_notes = list(by_note)
    rng.shuffle(active_notes)
    selected: List[Mapping[str, str]] = []
    while active_notes and len(selected) < limit:
        next_round: List[str] = []
        for note_id in active_notes:
            if len(selected) >= limit:
                break
            selected.append(by_note[note_id].pop())
            if by_note[note_id]:
                next_round.append(note_id)
        rng.shuffle(next_round)
        active_notes = next_round
    return selected


def write_sample(
    path: Path,
    rows: Iterable[Mapping[str, str]],
    input_fields: Sequence[str],
) -> None:
    _require_phi_safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields_without_review = [
        field for field in input_fields if field not in {"review_class", "correct"}
    ]
    if "note_id" in fields_without_review:
        fields_without_review.remove("note_id")
        fieldnames = ["note_id", "review_class", *fields_without_review, "correct"]
    else:
        fieldnames = ["review_class", *fields_without_review, "correct"]

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def _require_phi_safe_output(path: Path) -> None:
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
        rows, input_fields = read_report(args.input)
        sampled, counts = draw_balanced_sample(
            rows,
            per_class=args.per_class,
            seed=args.seed,
        )
        write_sample(args.output, sampled, input_fields)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Wrote {len(sampled)} review rows to {args.output}")
    for flag in ASSERTION_FLAGS:
        requested = args.per_class
        actual = counts[flag]
        suffix = "" if actual == requested else f" (only {actual} predicted positives available)"
        print(f"  {flag}: {actual}/{requested}{suffix}")


if __name__ == "__main__":
    main()
