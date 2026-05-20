# src/cohort/cohort_builder.py
"""
Cohort building pipeline for the MRSA NLP NER-based project.

Responsibilities
----------------
1. Load the existing MRSA cohort from mrsa_risk_predictions (same cohort, no rebuild).
2. Retrieve MRNs from CDMPHI.PERSON for all cohort persons.
3. Persist mrsa_cohort_person_list.parquet  (person_id | mrn | label).
4. Mine clinical notes from CDMPHI.NOTES in person-ID chunks and write one
   Parquet file per chunk to data/interim/airms/notes/.

The note-mining loop checks whether chunks already exist so it can be safely
re-run or resumed after a partial failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

LOG = logging.getLogger("mrsa_nlp.ner.cohort")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CohortConfig:
    """
    Tunable parameters for the cohort builder.

    Attributes
    ----------
    mrsa_predictions_interim_dir : Path
        Path to mrsa_risk_predictions/data/interim/airms/ (relative to
        project root or absolute).  Default resolves two directories above
        this project's root.
    cohort_parquet : str
        Filename inside mrsa_predictions_interim_dir that holds the final
        matched cohort (PERSON_ID + LABEL).
    schema : str
        HANA schema that owns the clinical data tables.
    notes_table : str
        Table name for clinical notes (typically NOTE in OMOP CDM).
    person_table : str
        Table name for person demographics (PERSON in OMOP CDM).
    note_out_dir : Path
        Local directory where note chunks are stored.
    cohort_person_list_path : Path
        Output path for the mrsa_cohort_person_list.parquet file.
    chunk_size : int
        Number of PERSON_IDs fetched per HANA query to limit memory usage.
    min_note_date : str
        Earliest NOTE_DATE to include (ISO format YYYY-MM-DD).
    note_type_concept_ids : list of int, optional
        OMOP NOTE_TYPE_CONCEPT_IDs to include.  None = all types.
    debug : bool
        When True limits mining to the first `debug_n_persons` persons.
    debug_n_persons : int
        How many persons to process in debug mode.
    """

    mrsa_predictions_interim_dir: Path = Path(
        "../../mrsa_risk_predictions/data/interim/airms"
    )
    cohort_parquet: str = "mrsa_visit_cohort.parquet"
    schema: str = "CDMPHI"
    notes_table: str = "NOTES"
    person_table: str = "PERSON"
    note_out_dir: Path = Path("data/interim/airms/notes")
    cohort_person_list_path: Path = Path(
        "data/interim/airms/mrsa_cohort_person_list.parquet"
    )
    chunk_size: int = 500
    min_note_date: str = "2014-07-14"
    note_type_concept_ids: Optional[List[int]] = None
    debug: bool = False
    debug_n_persons: int = 20


# ---------------------------------------------------------------------------
# Builder class
# ---------------------------------------------------------------------------

class CohortBuilder:
    """
    Orchestrates cohort creation and note mining for the NLP pipeline.

    Parameters
    ----------
    config : CohortConfig
        Configuration object with all tunable parameters.
    conn : hana_ml.dataframe.ConnectionContext
        Open HANA connection context.
    logger : logging.Logger, optional
        Logger to use; defaults to module-level LOG.

    Example
    -------
    >>> from src.cohort.cohort_builder import CohortConfig, CohortBuilder
    >>> from src.utils_db import connect_hana
    >>> cfg = CohortConfig(debug=True, debug_n_persons=10)
    >>> conn = connect_hana()
    >>> builder = CohortBuilder(cfg, conn)
    >>> person_df = builder.run()
    """

    def __init__(
        self,
        config: CohortConfig,
        conn,
        logger: logging.Logger = LOG,
    ) -> None:
        self.cfg = config
        self.conn = conn
        self.log = logger

    # ------------------------------------------------------------------
    # Step 1 — load existing cohort
    # ------------------------------------------------------------------

    def load_mrsa_predictions_cohort(self) -> pd.DataFrame:
        """
        Load the matched-pairs cohort produced by mrsa_risk_predictions.

        Reads ``{mrsa_predictions_interim_dir}/{cohort_parquet}`` and returns
        a deduplicated table with at minimum the columns PERSON_ID and LABEL.

        Returns
        -------
        pd.DataFrame
            Columns: PERSON_ID (int64), LABEL (int, 0=control 1=case).
            One row per unique person.

        Raises
        ------
        FileNotFoundError
            If the expected Parquet file does not exist.

        Notes
        -----
        - Use pd.read_parquet() to load the file.
        - After loading, keep only PERSON_ID and LABEL columns.
        - Deduplicate on PERSON_ID (keep max LABEL to preserve case status).
        - Log the total number of persons, cases, and controls.
        """
        pass

    # ------------------------------------------------------------------
    # Step 2 — look up MRNs
    # ------------------------------------------------------------------

    def get_person_mrns(self, person_ids: List[int]) -> pd.DataFrame:
        """
        Query CDMPHI.PERSON for MRNs (PERSON_SOURCE_VALUE) of the given persons.

        Parameters
        ----------
        person_ids : list of int
            PERSON_IDs to look up.

        Returns
        -------
        pd.DataFrame
            Columns: PERSON_ID (int64), MRN (str).

        Notes
        -----
        - Build an IN-clause from person_ids.  For large lists, split into
          batches of at most 1000 IDs (HANA has an IN-clause size limit).
        - Execute: SELECT PERSON_ID, PERSON_SOURCE_VALUE AS MRN
                   FROM {schema}.PERSON
                   WHERE PERSON_ID IN (...)
        - Convert PERSON_SOURCE_VALUE to string and strip whitespace.
        - Log the number of MRNs found vs expected.
        """
        pass

    # ------------------------------------------------------------------
    # Step 3 — save person list
    # ------------------------------------------------------------------

    def save_cohort_person_list(
        self,
        cohort_df: pd.DataFrame,
        mrn_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge cohort labels with MRNs and persist to disk.

        Parameters
        ----------
        cohort_df : pd.DataFrame
            PERSON_ID + LABEL (from load_mrsa_predictions_cohort).
        mrn_df : pd.DataFrame
            PERSON_ID + MRN (from get_person_mrns).

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with columns: PERSON_ID, MRN, LABEL.

        Notes
        -----
        - Left-join cohort_df on mrn_df using PERSON_ID.
        - Log how many persons are missing an MRN.
        - Log case count (LABEL==1) and control count (LABEL==0) separately
          so they can be compared with mrsa_risk_predictions numbers.
        - Write the result to cfg.cohort_person_list_path using write_parquet().
        - Also write a CSV version alongside the Parquet for quick inspection.
        """
        pass

    # ------------------------------------------------------------------
    # Step 4 — check if notes already exist
    # ------------------------------------------------------------------

    def check_notes_exist(self) -> bool:
        """
        Return True if at least one note chunk Parquet file is present in
        ``cfg.note_out_dir``.

        Returns
        -------
        bool
            True  → notes directory is non-empty (resume / skip mining).
            False → no notes found; mining is required.

        Notes
        -----
        - Check for files matching ``chunk_*.parquet`` inside note_out_dir.
        - Log the count of existing chunk files found.
        """
        pass

    # ------------------------------------------------------------------
    # Step 5 — mine notes in chunks
    # ------------------------------------------------------------------

    def _build_notes_query(
        self,
        person_ids_batch: List[int],
    ) -> str:
        """
        Build the SQL query that fetches notes for a batch of persons.

        Parameters
        ----------
        person_ids_batch : list of int
            PERSON_IDs for this chunk.

        Returns
        -------
        str
            SQL string ready to execute against the HANA connection.

        Notes
        -----
        SQL template (adapt column names to match the actual schema):

            SELECT
                N.NOTE_ID,
                N.PERSON_ID,
                N.NOTE_DATE,
                N.NOTE_DATETIME,
                N.NOTE_TYPE_CONCEPT_ID,
                N.NOTE_CLASS_CONCEPT_ID,
                N.NOTE_TITLE,
                N.NOTE_TEXT,
                N.VISIT_OCCURRENCE_ID
            FROM {schema}.{notes_table} N
            WHERE N.PERSON_ID IN ({ids})
              AND N.NOTE_DATE >= '{min_note_date}'
              [AND N.NOTE_TYPE_CONCEPT_ID IN ({type_ids})]  -- optional filter

        - Replace {ids} with a comma-separated list of person IDs.
        - Add the type_concept_id filter only when cfg.note_type_concept_ids
          is not None.
        """
        pass

    def mine_notes_chunked(self, person_ids: List[int]) -> None:
        """
        Query CDMPHI.NOTES for all persons in batches and save each batch as
        a Parquet chunk file under cfg.note_out_dir.

        Parameters
        ----------
        person_ids : list of int
            Full list of PERSON_IDs to mine notes for.

        Returns
        -------
        None
            Chunks are written to disk; nothing is returned.

        Notes
        -----
        Algorithm
        ~~~~~~~~~
        1. Create cfg.note_out_dir if it does not exist.
        2. Determine already-saved chunk indices from existing ``chunk_*.parquet``
           files so the loop can *skip* completed chunks on re-run.
        3. Split person_ids into sub-lists of size cfg.chunk_size.
        4. For each chunk i:
            a. If ``chunk_{i:04d}.parquet`` already exists → log skip, continue.
            b. Execute _build_notes_query() via the HANA connection.
            c. Fetch results into a pandas DataFrame.
            d. If the DataFrame is empty → log a warning, still write an empty
               file so the chunk index is marked as done.
            e. Write to ``note_out_dir/chunk_{i:04d}.parquet``.
            f. Log chunk index, person count, and note count.
        5. After the loop, log total chunks written and total notes collected.

        Robustness
        ~~~~~~~~~~
        - Wrap each chunk query in a try/except; on failure log the error and
          write a sentinel ``chunk_{i:04d}_FAILED.txt`` so you know which
          chunks need reprocessing without stopping the whole run.
        - Use tqdm for a progress bar when not in debug mode.
        """
        pass

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full cohort-building pipeline.

        Steps
        -----
        1. load_mrsa_predictions_cohort()    → cohort_df
        2. get_person_mrns(person_ids)       → mrn_df
        3. save_cohort_person_list(...)      → person_df
        4. check_notes_exist()
           - If True  → log "notes already exist, skipping mining"
           - If False → mine_notes_chunked(person_ids)

        Returns
        -------
        pd.DataFrame
            The mrsa_cohort_person_list DataFrame (PERSON_ID, MRN, LABEL).

        Notes
        -----
        - In debug mode, restrict person_ids to the first cfg.debug_n_persons
          entries and log clearly that this is a debug run.
        - Log timing at the start and end of run().
        """
        pass
