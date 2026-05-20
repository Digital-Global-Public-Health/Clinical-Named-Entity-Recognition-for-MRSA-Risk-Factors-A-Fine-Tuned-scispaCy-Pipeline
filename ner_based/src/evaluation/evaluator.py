# src/evaluation/evaluator.py
"""
Evaluation and visualisation pipeline for the NER-based extraction.

Computes entity-level NER metrics (precision/recall/F1 per entity type) against
a gold-standard annotation set, and produces descriptive analysis of the feature
matrix (prevalence, coverage, comparison against rule-based features).

Input  : outputs/<run_dir>/ner_features_*.csv
         annotations/test.spacy  (or test.csv gold standard)
Output : outputs/<run_dir>/evaluation/
             ner_metrics_by_entity.csv
             ner_metrics_by_entity.png
             training_f1_curve.png
             feature_prevalence.png
             ner_vs_rules_comparison.csv  (optional, if rule features provided)
             validation_report.txt
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

LOG = logging.getLogger("mrsa_nlp.ner.evaluation")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NEREvaluatorConfig:
    """
    Parameters for the NER evaluator.

    Attributes
    ----------
    features_path : Path
        Path to the NER feature matrix CSV.
    test_annotations_path : Path, optional
        Path to the held-out annotated test set (.spacy or CSV).
        Required for entity-level precision/recall/F1.
    rule_features_path : Path, optional
        Path to the rule-based feature matrix CSV for comparison analysis.
    out_dir : Path
        Sub-directory within run_dir for evaluation outputs.
    entity_labels : list of str
        Entity types to evaluate.
    target_f1 : float
        Minimum F1 score per entity type (pass/fail threshold).
    n_example_notes : int
        Number of example predictions to include in the report.
    plot_dpi : int
        DPI for saved figures.
    debug : bool
    """

    features_path: Path = Path("outputs/ner_features.csv")
    test_annotations_path: Optional[Path] = None
    rule_features_path: Optional[Path] = None
    out_dir: Path = Path("outputs/evaluation")
    entity_labels: List[str] = field(
        default_factory=lambda: ["DISEASE", "MEDICATION", "PROCEDURE"]
    )
    target_f1: float = 0.70
    n_example_notes: int = 10
    plot_dpi: int = 150
    debug: bool = False


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class NEREvaluator:
    """
    Evaluates NER extraction quality and generates visual reports.

    Parameters
    ----------
    config : NEREvaluatorConfig
    run_dir : Path
    logger : logging.Logger, optional

    Example
    -------
    >>> from src.evaluation.evaluator import NEREvaluatorConfig, NEREvaluator
    >>> cfg = NEREvaluatorConfig(features_path=Path("outputs/.../ner_features.csv"))
    >>> evaluator = NEREvaluator(cfg, run_dir=Path("outputs/ner_eval_run"))
    >>> evaluator.run()
    """

    def __init__(
        self,
        config: NEREvaluatorConfig,
        run_dir: Path,
        logger: logging.Logger = LOG,
    ) -> None:
        self.cfg = config
        self.run_dir = run_dir
        self.eval_dir = run_dir / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.log = logger

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_features(self) -> pd.DataFrame:
        """
        Load the NER feature matrix.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FileNotFoundError
        """
        pass

    def load_test_annotations(self) -> Optional[Any]:
        """
        Load the held-out test annotations for entity-level evaluation.

        Returns
        -------
        list of spaCy Example or structured dict, or None if not configured.

        Notes
        -----
        - Supports .spacy (DocBin) and CSV formats.
        - CSV format: note_id, text_segment, entity, entity_type.
        """
        pass

    def load_rule_features(self) -> Optional[pd.DataFrame]:
        """
        Load rule-based features for comparison (optional).

        Returns
        -------
        pd.DataFrame or None
        """
        pass

    # ------------------------------------------------------------------
    # Entity-level NER metrics
    # ------------------------------------------------------------------

    def compute_ner_metrics(
        self,
        predictions: List[List[str]],
        gold_labels: List[List[str]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute token-level precision/recall/F1 per entity type.

        Uses seqeval for span-level evaluation (strict matching).

        Parameters
        ----------
        predictions : list of list of str
            Predicted BIO tag sequences.
        gold_labels : list of list of str
            Gold BIO tag sequences.

        Returns
        -------
        dict of {entity_type: {precision, recall, f1, support}}

        Notes
        -----
        - Use ``seqeval.metrics.classification_report()`` for the full report.
        - Also compute micro-averaged metrics overall.
        - Log a formatted metrics table to the console.
        """
        pass

    def run_model_evaluation(
        self,
        model,
        test_examples: Any,
    ) -> Dict[str, float]:
        """
        Run the trained NER model on the test set and compute metrics.

        Parameters
        ----------
        model : spacy.Language or HuggingFace model
        test_examples : list of spaCy Examples or structured data

        Returns
        -------
        dict of {str: float}
            Entity-level and overall metrics.

        Notes
        -----
        - For spaCy: ``nlp.evaluate(examples).scores``.
        - For HuggingFace: run inference, decode BIO tags, pass to
          compute_ner_metrics().
        """
        pass

    # ------------------------------------------------------------------
    # Feature-level analysis (no model needed)
    # ------------------------------------------------------------------

    def compute_prevalence(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute NER feature prevalence overall and by LABEL.

        Parameters
        ----------
        features_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Columns: entity_label, feature_type, prevalence_overall,
            prevalence_cases, prevalence_controls, prevalence_ratio.

        Notes
        -----
        - Separate rows for has_*, count_*, and has_*_negated features.
        - Log the top-5 most prevalent entity features.
        """
        pass

    def compare_with_rules(
        self,
        ner_features_df: pd.DataFrame,
        rule_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Agreement analysis between NER and rule-based features.

        Parameters
        ----------
        ner_features_df : pd.DataFrame
        rule_features_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Columns: feature, agreement_rate, ner_only_rate,
            rules_only_rate, both_rate, neither_rate.

        Notes
        -----
        - Join on PERSON_ID or VISIT_OCCURRENCE_ID.
        - For each common has_* feature, compute percentage of visits where:
            * Both agree positive
            * Both agree negative
            * NER positive, rules negative (NER found extra)
            * NER negative, rules positive (rules found extra)
        - Log the top features where NER and rules disagree most.
        """
        pass

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_entity_metrics(
        self,
        metrics: Dict[str, Dict[str, float]],
    ) -> Path:
        """
        Grouped bar chart of precision/recall/F1 per entity type.

        Parameters
        ----------
        metrics : dict

        Returns
        -------
        Path
            Saved figure: ``eval_dir/ner_metrics_by_entity.png``.

        Notes
        -----
        - Horizontal dashed line at cfg.target_f1.
        - Colour-code bars: green if F1 ≥ target, red if below.
        """
        pass

    def plot_prevalence(
        self,
        prevalence_df: pd.DataFrame,
    ) -> Path:
        """
        Stacked / grouped bar chart of NER feature prevalence by LABEL.

        Returns
        -------
        Path
            ``eval_dir/feature_prevalence.png``.
        """
        pass

    def plot_ner_vs_rules(
        self,
        comparison_df: pd.DataFrame,
    ) -> Path:
        """
        Stacked bar chart showing NER vs rules agreement breakdown per feature.

        Returns
        -------
        Path
            ``eval_dir/ner_vs_rules_comparison.png``.
        """
        pass

    def plot_label_distribution(
        self,
        features_df: pd.DataFrame,
    ) -> Path:
        """
        Bar / pie chart of LABEL distribution in the feature matrix.

        Returns
        -------
        Path
            ``eval_dir/label_distribution.png``.
        """
        pass

    # ------------------------------------------------------------------
    # Validation report
    # ------------------------------------------------------------------

    def generate_validation_report(
        self,
        features_df: pd.DataFrame,
        ner_metrics: Optional[Dict],
        prevalence_df: pd.DataFrame,
        comparison_df: Optional[pd.DataFrame],
    ) -> Path:
        """
        Write a structured plain-text validation report.

        Sections
        --------
        1. Header: date, dataset size, label distribution.
        2. NER entity metrics table (if test annotations available).
        3. Feature prevalence table.
        4. NER vs rule-based comparison (if rule features available).
        5. Pass/fail summary vs cfg.target_f1.
        6. Limitations and recommended next steps.

        Returns
        -------
        Path
            ``eval_dir/validation_report.txt``.
        """
        pass

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full NER evaluation pipeline.

        Steps
        -----
        1. load_features()
        2. compute_prevalence()  → plot_prevalence(), plot_label_distribution()
        3. If test annotations available:
            a. load_test_annotations()
            b. run_model_evaluation() (requires model loading externally)
            c. compute_ner_metrics()
            d. plot_entity_metrics()
        4. If rule features available:
            a. load_rule_features()
            b. compare_with_rules()
            c. plot_ner_vs_rules()
        5. generate_validation_report()

        Notes
        -----
        - Log a final pass/fail summary to console.
        - All outputs go to self.eval_dir.
        """
        pass
