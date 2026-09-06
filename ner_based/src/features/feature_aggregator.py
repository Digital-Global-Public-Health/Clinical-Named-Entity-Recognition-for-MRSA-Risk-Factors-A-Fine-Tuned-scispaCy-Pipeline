# src/features/feature_aggregator.py
"""
NER feature engineering and aggregation pipeline.

Aggregates per-note NER extraction results to visit-level features and merges
with the MRSA cohort labels to produce the final training-ready matrix.

Input  : data/interim/airms/ner_extractions/chunk_*.parquet
         data/interim/airms/mrsa_cohort_person_list.parquet
Output : outputs/<run_dir>/ner_features_<timestamp>.csv
         outputs/<run_dir>/ner_feature_summary_<timestamp>.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LOG = logging.getLogger("mrsa_nlp.ner.features")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NERAggregatorConfig:
    """
    Parameters for the NER feature aggregator.

    Attributes
    ----------
    extractions_dir : Path
        Directory with NER extraction chunk Parquet files.
    cohort_person_list_path : Path
        Path to mrsa_cohort_person_list.parquet.
    out_dir : Path
        Base output directory.
    aggregation_level : str
        "visit" (primary) or "person".
    entity_labels : list of str
        Entity types to produce features for.
    include_negated_features : bool
        Include has_{label}_negated columns for negated entity mentions.
    include_entity_counts : bool
        Include count_{label} columns.
    include_note_type_breakdown : bool
        Produce features split by NOTE_TYPE_CONCEPT_ID.
    lookback_days : int
        Parameterized lookback window for future leakage-gate filtering.
    fill_missing_with_zero : bool
        Fill NaN values with 0 after cohort merge.
    debug : bool
    debug_n_extractions : int
    """

    extractions_dir: Path = Path("data/interim/airms/ner_extractions")
    cohort_person_list_path: Path = Path(
        "data/interim/airms/mrsa_cohort_person_list.parquet"
    )
    out_dir: Path = Path("outputs")
    aggregation_level: str = "visit"
    entity_labels: List[str] = field(
        default_factory=lambda: ["DISEASE", "MEDICATION", "PROCEDURE"]
    )
    include_negated_features: bool = True
    include_entity_counts: bool = True
    include_note_type_breakdown: bool = False
    lookback_days: int = 90
    fill_missing_with_zero: bool = True
    debug: bool = False
    debug_n_extractions: int = 1000


# ---------------------------------------------------------------------------
# Aggregator class
# ---------------------------------------------------------------------------

class NERFeatureAggregator:
    """
    Transforms per-note NER extraction results into a visit-level feature matrix.

    Parameters
    ----------
    config : NERAggregatorConfig
    run_dir : Path
    logger : logging.Logger, optional

    Example
    -------
    >>> from src.features.feature_aggregator import NERAggregatorConfig, NERFeatureAggregator
    >>> cfg = NERAggregatorConfig(debug=True)
    >>> agg = NERFeatureAggregator(cfg, run_dir=Path("outputs/ner_features_run"))
    >>> feature_df = agg.run()
    """

    def __init__(
        self,
        config: NERAggregatorConfig,
        run_dir: Path,
        logger: logging.Logger = LOG,
    ) -> None:
        self.cfg = config
        self.run_dir = run_dir
        self.log = logger

    def apply_leakage_gate_filter(self, extractions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter note/entity rows to the configured pre-index lookback window.

        TODO: this needs the index date/index event definition from the supervisor
        before it can be implemented without introducing target leakage.
        """
        raise NotImplementedError("blocked: needs index-event definition from supervisor")

    def load_extractions(self) -> pd.DataFrame:
        """
        Load and concatenate all NER extraction chunk Parquet files.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with NER feature columns.

        Raises
        ------
        FileNotFoundError
            If cfg.extractions_dir is empty. Prompt user to run extraction
            pipeline first.
        """
        pass

    def load_cohort(self) -> pd.DataFrame:
        """
        Load mrsa_cohort_person_list.

        Returns
        -------
        pd.DataFrame
            Columns: PERSON_ID, MRN, LABEL.
        """
        pass

    def _get_feature_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Identify has_*, count_*, and has_*_negated columns in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        Tuple[list, list, list]
            (binary_cols, count_cols, negated_cols)
        """
        pass

    def aggregate_to_visit_level(self, extractions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-note NER features to one row per VISIT_OCCURRENCE_ID.

        Aggregation:
        - Binary features (has_*): MAX across notes (1 if any note has entity).
        - Count features (count_*): SUM across notes.
        - Negated features (has_*_negated): MAX.

        Parameters
        ----------
        extractions_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Columns: VISIT_OCCURRENCE_ID, PERSON_ID, n_notes, feature_cols*.

        Notes
        -----
        - Also include n_notes_in_visit (note count per visit).
        - Log number of visits with at least one entity detected.
        """
        pass

    def aggregate_to_person_level(self, extractions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-note NER features to one row per PERSON_ID.

        Parameters
        ----------
        extractions_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        pass

    def merge_with_cohort(
        self,
        features_df: pd.DataFrame,
        cohort_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join feature matrix with cohort labels on PERSON_ID.

        Parameters
        ----------
        features_df : pd.DataFrame
        cohort_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Features + LABEL + MRN.

        Notes
        -----
        - Log final case/control counts.
        - Fill NaN features with 0 if cfg.fill_missing_with_zero.
        """
        pass

    def compute_feature_summary(self, feature_df: pd.DataFrame) -> Dict:
        """
        Compute descriptive statistics per NER feature.

        Parameters
        ----------
        feature_df : pd.DataFrame

        Returns
        -------
        dict
            Prevalence and count statistics per entity type and label.
        """
        pass

    def export(
        self,
        feature_df: pd.DataFrame,
        summary: Dict,
        timestamp: str,
    ) -> Path:
        """
        Persist the feature matrix and summary JSON.

        Parameters
        ----------
        feature_df : pd.DataFrame
        summary : dict
        timestamp : str

        Returns
        -------
        Path
            Path to the saved CSV.

        Notes
        -----
        - CSV:     ``{run_dir}/ner_features_{timestamp}.csv``
        - Parquet: ``{run_dir}/ner_features_{timestamp}.parquet``
        - JSON:    ``{run_dir}/ner_feature_summary_{timestamp}.json``
        """
        pass

    def run(self) -> pd.DataFrame:
        """
        Execute the full NER feature aggregation pipeline.

        Steps
        -----
        1. load_extractions()
        2. load_cohort()
        3. aggregate by level
        4. merge_with_cohort()
        5. compute_feature_summary()
        6. export()

        Returns
        -------
        pd.DataFrame
        """
        pass
