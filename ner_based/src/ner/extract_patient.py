"""Extract PHI-bearing notes for selected patients without loading the cohort."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from pyarrow import ArrowException as ArrowError
except ImportError:  # pragma: no cover - compatibility with older pyarrow
    from pyarrow import ArrowInvalid as ArrowError

LOG = logging.getLogger("mrsa_nlp.ner.extract_patient")

DEFAULT_NOTE_TITLES: Tuple[str, ...] = (
    "Progress Notes",
    "Consults",
    "H&P",
    "Discharge Summary",
)
DEFAULT_OUTPUT_PATH = Path(
    "/sc/arion/work/rademt02/airms_notes/extracted/"
    "extracted_patient_notes.parquet"
)

REQUIRED_NOTE_COLUMNS: Tuple[str, ...] = (
    "NOTE_ID",
    "PERSON_ID",
    "NOTE_TEXT",
    "NOTE_TITLE",
)
OPTIONAL_NOTE_COLUMNS: Tuple[str, ...] = (
    "NOTE_DATETIME",
    "VISIT_OCCURRENCE_ID",
)


class PatientExtractionError(ValueError):
    """Raised when a requested patient extraction cannot be completed safely."""


@dataclass(frozen=True)
class PatientNoteExtractionConfig:
    notes_parquet: Path
    cohort_csv: Path
    person_ids: Tuple[object, ...]
    note_titles: Tuple[str, ...] = DEFAULT_NOTE_TITLES
    output: Path = DEFAULT_OUTPUT_PATH
    require_cohort_membership: bool = True


@dataclass(frozen=True)
class ExtractionSummary:
    total_notes: int
    counts_by_title: Dict[str, int]
    output: Path


def parse_note_titles(value: str) -> Tuple[str, ...]:
    """Parse a comma-separated title option while preserving input order."""
    titles = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not titles:
        raise PatientExtractionError("--note-titles must contain at least one title")
    return titles


def _coerce_person_id(value: object, arrow_type: pa.DataType) -> object:
    """Coerce an external ID to the parquet field's scalar type."""
    raw = str(value).strip()
    if not raw:
        raise PatientExtractionError(
            "--person-id is incompatible with the parquet PERSON_ID type"
        )

    try:
        if pa.types.is_integer(arrow_type):
            return pa.scalar(int(raw), type=arrow_type).as_py()
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return pa.scalar(raw, type=arrow_type).as_py()
    except (ArrowError, OverflowError, TypeError, ValueError) as exc:
        raise PatientExtractionError(
            f"--person-id is incompatible with parquet PERSON_ID type {arrow_type}"
        ) from exc

    raise PatientExtractionError(
        f"unsupported parquet PERSON_ID type {arrow_type}; expected integer or string"
    )


def _normalize_cohort_ids(
    cohort: pd.DataFrame,
    arrow_type: pa.DataType,
) -> pd.DataFrame:
    """Normalize CSV IDs to the parquet type and discard blank identifier rows."""
    normalized_ids = []
    retained_indices = []

    for index, value in cohort["PERSON_ID"].items():
        if pd.isna(value) or not str(value).strip():
            continue
        try:
            normalized = _coerce_person_id(value, arrow_type)
        except PatientExtractionError as exc:
            raise PatientExtractionError(
                f"cohort PERSON_ID is incompatible with parquet PERSON_ID type {arrow_type}"
            ) from exc
        retained_indices.append(index)
        normalized_ids.append(normalized)

    normalized = cohort.loc[retained_indices].copy()
    normalized["PERSON_ID"] = pd.Series(
        normalized_ids,
        index=retained_indices,
    )
    return normalized


def _person_filter(person_ids: Sequence[object]) -> list[tuple[str, str, object]]:
    if len(person_ids) == 1:
        return [("PERSON_ID", "==", person_ids[0])]
    return [("PERSON_ID", "in", list(person_ids))]


def extract_patient_notes(
    config: PatientNoteExtractionConfig,
    logger: logging.Logger = LOG,
) -> ExtractionSummary:
    """Extract selected patients' full notes with read-time predicate pushdown."""
    if not config.person_ids:
        raise PatientExtractionError("at least one --person-id is required")
    if not config.note_titles:
        raise PatientExtractionError("--note-titles must contain at least one title")

    parquet_file = pq.ParquetFile(config.notes_parquet)
    schema = parquet_file.schema_arrow
    available_columns = set(schema.names)
    missing_columns = [
        column for column in REQUIRED_NOTE_COLUMNS if column not in available_columns
    ]
    if missing_columns:
        raise PatientExtractionError(
            "notes parquet is missing required columns: " + ", ".join(missing_columns)
        )

    person_id_type = schema.field("PERSON_ID").type
    typed_person_ids = tuple(
        _coerce_person_id(person_id, person_id_type)
        for person_id in config.person_ids
    )

    cohort = pd.read_csv(
        config.cohort_csv,
        usecols=["PERSON_ID", "LABEL"],
        dtype={"PERSON_ID": "string"},
    )
    cohort = _normalize_cohort_ids(cohort, person_id_type)
    cohort = cohort.drop_duplicates("PERSON_ID")
    cohort_ids = set(cohort["PERSON_ID"].tolist())
    missing_from_cohort = [
        person_id for person_id in typed_person_ids if person_id not in cohort_ids
    ]
    if config.require_cohort_membership and missing_from_cohort:
        raise PatientExtractionError("patient not in cohort")

    projected_columns = list(REQUIRED_NOTE_COLUMNS)
    projected_columns.extend(
        column for column in OPTIONAL_NOTE_COLUMNS if column in available_columns
    )
    notes = pd.read_parquet(
        config.notes_parquet,
        filters=_person_filter(typed_person_ids),
        columns=projected_columns,
        engine="pyarrow",
    )

    notes = notes.loc[notes["NOTE_TITLE"].isin(config.note_titles)].copy()
    text_present = (
        notes["NOTE_TEXT"].notna()
        & notes["NOTE_TEXT"].astype("string").str.strip().ne("")
    )
    notes = notes.loc[text_present].copy()
    if notes.empty:
        raise PatientExtractionError(
            "patient in cohort but no notes of requested titles"
        )

    duplicate_note_rows = int(notes["NOTE_ID"].duplicated(keep=False).sum())
    if duplicate_note_rows:
        raise PatientExtractionError(
            f"duplicate NOTE_ID values after filtering: {duplicate_note_rows} rows"
        )

    selected_cohort = cohort.loc[
        cohort["PERSON_ID"].isin(typed_person_ids),
        ["PERSON_ID", "LABEL"],
    ]
    notes = notes.merge(
        selected_cohort,
        on="PERSON_ID",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    sort_columns = [
        column for column in ("NOTE_DATETIME", "NOTE_ID") if column in notes.columns
    ]
    notes = notes.sort_values(
        sort_columns,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    output_columns = [
        "NOTE_ID",
        "PERSON_ID",
        "NOTE_TEXT",
        "NOTE_TITLE",
        "LABEL",
    ]
    output_columns.extend(
        column for column in OPTIONAL_NOTE_COLUMNS if column in notes.columns
    )
    notes = notes.loc[:, output_columns]

    if config.output.exists() and config.output.is_dir():
        raise PatientExtractionError("output path is an existing directory")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    notes.to_parquet(config.output, index=False, engine="pyarrow")

    counts = {
        title: int(notes["NOTE_TITLE"].eq(title).sum())
        for title in config.note_titles
    }
    logger.info("Total notes extracted: %d", len(notes))
    for title, count in counts.items():
        logger.info("%s: %d", title, count)

    return ExtractionSummary(
        total_notes=len(notes),
        counts_by_title=counts,
        output=config.output,
    )
