# src/ner/model_trainer.py
"""
NER model training and fine-tuning pipeline.

Supports two training tracks:
  Track A — scispaCy fine-tuning
      Fine-tune the pre-trained en_core_sci_sm model on annotated clinical
      notes using spaCy's DocBin + training loop.

  Track B — BioClinicalBERT fine-tuning (advanced)
      Fine-tune emilyalsentzer/Bio_ClinicalBERT with a token-classification
      head using HuggingFace Transformers.  Requires GPU (Minerva).

Input  : annotations/train.spacy  (or annotations/train.csv)
         annotations/val.spacy    (or annotations/val.csv)
Output : models/airms_ner_<version>/
             config.cfg  (spaCy) or config.yaml (HF)
             model artifacts
         outputs/<run_dir>/training_curves.png
         outputs/<run_dir>/metrics.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("mrsa_nlp.ner.trainer")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NERTrainerConfig:
    """
    Tunable parameters for NER model training.

    Attributes
    ----------
    track : str
        "spacy" for scispaCy fine-tuning or "hf" for HuggingFace/BERT.
    base_model : str
        spaCy: "en_core_sci_sm" | "en_core_sci_lg".
        HF: "emilyalsentzer/Bio_ClinicalBERT" | "allenai/biomed_roberta_base".
    entity_labels : list of str
        NER labels to train (must match annotation schema).
    train_data_path : Path
        Path to training annotations (spaCy DocBin or CSV).
    val_data_path : Path
        Path to validation annotations.
    test_data_path : Path
        Path to held-out test annotations (evaluated only at the end).
    model_out_dir : Path
        Directory where model checkpoints and final model are saved.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for the training loop.
    dropout : float
        Dropout rate during training.
    learning_rate : float
        Initial learning rate.
    eval_every_n_epochs : int
        How often to evaluate on the validation set during training.
    min_f1_to_save : float
        Minimum overall F1 on validation to save a checkpoint.
    early_stopping_patience : int
        Stop training if validation F1 does not improve for this many epochs.
    device : str
        "cpu" or "gpu" (or "cuda:0" for HF).
    seed : int
        Random seed for reproducibility.
    debug : bool
        Limit training to a small subset for quick iteration.
    debug_n_examples : int
        Max training examples in debug mode.
    """

    track: str = "spacy"
    base_model: str = "en_core_sci_sm"
    entity_labels: List[str] = field(
        default_factory=lambda: ["DISEASE", "MEDICATION", "PROCEDURE"]
    )
    train_data_path: Path = Path("annotations/train.spacy")
    val_data_path: Path = Path("annotations/val.spacy")
    test_data_path: Path = Path("annotations/test.spacy")
    model_out_dir: Path = Path("models/airms_ner_v1.0")
    n_epochs: int = 30
    batch_size: int = 16
    dropout: float = 0.3
    learning_rate: float = 1e-3
    eval_every_n_epochs: int = 5
    min_f1_to_save: float = 0.60
    early_stopping_patience: int = 10
    device: str = "cpu"
    seed: int = 7  # Global seed for reproducibility (matches mrsa_risk_predictions)
    debug: bool = False
    debug_n_examples: int = 50


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class NERModelTrainer:
    """
    Trains or fine-tunes a NER model for MRSA clinical note entity extraction.

    Supports scispaCy (Track A) and HuggingFace BERT (Track B).

    Parameters
    ----------
    config : NERTrainerConfig
    schema : AnnotationSchema
        Schema defining entity labels.
    run_dir : Path
        Timestamped output directory for this training run.
    logger : logging.Logger, optional

    Example
    -------
    >>> from src.ner.model_trainer import NERTrainerConfig, NERModelTrainer
    >>> from src.ner.annotation_schema import AnnotationSchema, AnnotationSchemaConfig
    >>> schema = AnnotationSchema(AnnotationSchemaConfig())
    >>> cfg = NERTrainerConfig(debug=True)
    >>> trainer = NERModelTrainer(cfg, schema, run_dir=Path("outputs/train_run"))
    >>> model = trainer.run()
    """

    def __init__(
        self,
        config: NERTrainerConfig,
        schema,
        run_dir: Path,
        logger: logging.Logger = LOG,
    ) -> None:
        self.cfg = config
        self.schema = schema
        self.run_dir = run_dir
        self.log = logger

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_training_data(self) -> Tuple[Any, Any]:
        """
        Load training and validation annotation data.

        Returns
        -------
        Tuple[train_data, val_data]
            For spaCy track: (list of spaCy Example objects, ...).
            For HF track: (datasets.Dataset, datasets.Dataset).

        Raises
        ------
        FileNotFoundError
            If cfg.train_data_path or cfg.val_data_path does not exist.

        Notes
        -----
        Track A (spaCy):
          - Load via ``spacy.tokens.DocBin().from_disk()`` for .spacy format.
          - Or parse CSV annotation format with columns:
            note_id, text_segment, entity, entity_type.
          - Convert to spaCy ``Example`` objects using a blank or base NLP.

        Track B (HF):
          - Load via ``datasets.load_from_disk()`` or custom CSV parser.
          - Ensure BIO-tagged token sequences are produced.
          - Apply tokenization with the HuggingFace tokenizer.

        In debug mode, truncate to cfg.debug_n_examples.
        """
        pass

    # ------------------------------------------------------------------
    # Track A — spaCy fine-tuning
    # ------------------------------------------------------------------

    def load_spacy_base_model(self):
        """
        Load the pre-trained scispaCy base model specified in cfg.base_model.

        Returns
        -------
        spacy.Language
            The loaded NLP pipeline.

        Raises
        ------
        OSError
            If the model package is not installed.

        Notes
        -----
        - Use ``spacy.load(cfg.base_model)``.
        - If the NER component is not present, add a blank one:
          ``nlp.add_pipe("ner")``.
        """
        pass

    def add_entity_labels(self, nlp) -> None:
        """
        Register all entity labels from cfg.entity_labels with the NER component.

        Parameters
        ----------
        nlp : spacy.Language
            The loaded spaCy pipeline.

        Notes
        -----
        - Call ``ner.add_label(label)`` for each label not already registered.
        - Log which labels were added vs already present.
        """
        pass

    def train_spacy_epoch(
        self,
        nlp,
        optimizer,
        examples: List,
        drop: float,
    ) -> Dict[str, float]:
        """
        Run one training epoch over the spaCy examples.

        Parameters
        ----------
        nlp : spacy.Language
        optimizer : Optimizer
            spaCy optimizer returned by ``nlp.resume_training()``.
        examples : list
            Shuffled list of spaCy Example objects.
        drop : float
            Dropout probability.

        Returns
        -------
        dict of {str: float}
            Training loss values per component (e.g. {"ner": 0.45}).

        Notes
        -----
        - Use ``spacy.util.minibatch(examples, size=cfg.batch_size)`` for batching.
        - Call ``nlp.update(batch, sgd=optimizer, drop=drop, losses=losses)``.
        - Shuffle examples at the start of each epoch.
        """
        pass

    def evaluate_spacy(
        self,
        nlp,
        examples: List,
    ) -> Dict[str, float]:
        """
        Evaluate the spaCy NER model on a set of examples.

        Parameters
        ----------
        nlp : spacy.Language
        examples : list of spaCy Example objects

        Returns
        -------
        dict of {str: float}
            Keys: "precision", "recall", "f1" (overall).
            Also includes per-entity-type "f1_{entity_type}".

        Notes
        -----
        - Use ``nlp.evaluate(examples)`` and extract ``.scores["ents_f"]`` etc.
        - Log per-entity-type scores for diagnostic purposes.
        """
        pass

    def train_spacy(
        self,
        nlp,
        train_examples: List,
        val_examples: List,
    ) -> Tuple[Any, Dict]:
        """
        Run the full spaCy fine-tuning loop.

        Parameters
        ----------
        nlp : spacy.Language
        train_examples : list
        val_examples : list

        Returns
        -------
        Tuple[spacy.Language, dict]
            Best model and its validation metrics.

        Notes
        -----
        Algorithm
        ~~~~~~~~~
        1. ``optimizer = nlp.resume_training()``  (or ``nlp.begin_training()``).
        2. Track best_f1 = 0, patience_count = 0.
        3. For epoch in range(cfg.n_epochs):
            a. train_spacy_epoch()
            b. If epoch % cfg.eval_every_n_epochs == 0:
                 metrics = evaluate_spacy(val_examples)
                 Log epoch, loss, val F1.
                 If metrics["f1"] > best_f1 and > cfg.min_f1_to_save:
                     save_checkpoint(); best_f1 = metrics["f1"]
                 Else: patience_count += 1
                 If patience_count >= cfg.early_stopping_patience: break
        4. Plot training curves.
        5. Return best checkpoint.
        """
        pass

    # ------------------------------------------------------------------
    # Track B — HuggingFace BERT fine-tuning
    # ------------------------------------------------------------------

    def load_hf_model(self) -> Tuple[Any, Any]:
        """
        Load the HuggingFace tokenizer and token-classification model.

        Returns
        -------
        Tuple[tokenizer, model]
            HuggingFace AutoTokenizer and AutoModelForTokenClassification
            initialised from cfg.base_model with cfg.entity_labels as labels.

        Notes
        -----
        - Use ``transformers.AutoTokenizer.from_pretrained(cfg.base_model)``.
        - Build label2id / id2label mappings from cfg.entity_labels using
          BIO tagging: B-DISEASE, I-DISEASE, B-MEDICATION, ..., O.
        - Move model to cfg.device.
        """
        pass

    def train_hf(
        self,
        tokenizer,
        model,
        train_dataset,
        val_dataset,
    ) -> Tuple[Any, Dict]:
        """
        Fine-tune the HuggingFace token-classification model.

        Parameters
        ----------
        tokenizer : HuggingFace tokenizer
        model : AutoModelForTokenClassification
        train_dataset : datasets.Dataset
        val_dataset : datasets.Dataset

        Returns
        -------
        Tuple[model, dict]
            Best model and validation metrics.

        Notes
        -----
        - Use ``transformers.Trainer`` or a custom training loop.
        - Track per-epoch F1 using ``seqeval`` metric.
        - Apply early stopping (``EarlyStoppingCallback``).
        - Save checkpoints to cfg.model_out_dir / "hf_checkpoints".
        """
        pass

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        model,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Path:
        """
        Save a model checkpoint to disk.

        Parameters
        ----------
        model
            spaCy Language or HuggingFace model.
        epoch : int
            Current epoch number (used in checkpoint directory name).
        metrics : dict
            Validation metrics to log alongside the checkpoint.

        Returns
        -------
        Path
            Checkpoint directory path.

        Notes
        -----
        - spaCy: ``nlp.to_disk(path)``.
        - HF: ``model.save_pretrained(path)``; tokenizer too.
        - Write metrics.json to checkpoint directory.
        """
        pass

    def load_best_checkpoint(self) -> Any:
        """
        Load the best saved model checkpoint from cfg.model_out_dir.

        Returns
        -------
        spacy.Language or HuggingFace model
            The best model based on validation F1.

        Notes
        -----
        - Read checkpoint metadata (metrics.json files) to identify the
          checkpoint with the highest F1.
        - Load and return that model.
        """
        pass

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_f1_scores: List[float],
        eval_epochs: List[int],
    ) -> Path:
        """
        Save a training/validation curve plot.

        Parameters
        ----------
        train_losses : list of float
            Per-epoch training loss.
        val_f1_scores : list of float
            Validation F1 at each evaluation epoch.
        eval_epochs : list of int
            Epoch indices at which validation was run.

        Returns
        -------
        Path
            Saved figure: ``run_dir/training_curves.png``.

        Notes
        -----
        - Twin-axis plot: loss (left) and F1 (right).
        - Mark the best-F1 epoch with a vertical dashed line.
        """
        pass

    def evaluate_on_test(self, model) -> Dict[str, float]:
        """
        Run final evaluation on the held-out test set.

        Parameters
        ----------
        model : spacy.Language or HuggingFace model

        Returns
        -------
        dict of {str: float}
            Precision, recall, F1 overall and per entity type.

        Notes
        -----
        - Load test data from cfg.test_data_path.
        - Log and save results to ``run_dir/test_metrics.json``.
        """
        pass

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> Any:
        """
        Execute the full training pipeline.

        Steps
        -----
        1. load_training_data()            → train_data, val_data
        2. Dispatch on cfg.track:
           Track A: load_spacy_base_model() → add_entity_labels() → train_spacy()
           Track B: load_hf_model()         → train_hf()
        3. evaluate_on_test()
        4. plot_training_curves()
        5. Save final model to cfg.model_out_dir.
        6. Write run_dir/metrics.json with all evaluation results.

        Returns
        -------
        model
            The best trained model object.
        """
        pass
