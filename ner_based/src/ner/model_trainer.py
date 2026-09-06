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
import random
import csv
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
        if self.cfg.track.lower() != "spacy":
            raise NotImplementedError("Track B / HuggingFace training is out of thesis scope")

        nlp = getattr(self, "_loading_nlp", None)
        if nlp is None:
            import spacy

            nlp = spacy.blank("en")

        train_docs = self._load_spacy_docbin(self.cfg.train_data_path, nlp.vocab)
        val_docs = self._load_spacy_docbin(self.cfg.val_data_path, nlp.vocab)

        if self.cfg.debug:
            train_docs = train_docs[: self.cfg.debug_n_examples]
            val_docs = val_docs[: self.cfg.debug_n_examples]
            self.log.info(
                "Debug mode: using %d train docs and %d validation docs",
                len(train_docs),
                len(val_docs),
            )

        return train_docs, val_docs

    def _load_spacy_docbin(self, path: Path, vocab) -> List[Any]:
        """Load one .spacy DocBin and validate the note-level contract."""
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")
        if path.suffix != ".spacy":
            raise ValueError(f"Only .spacy DocBin inputs are implemented for Track A: {path}")

        from spacy.tokens import DocBin

        docs = list(DocBin().from_disk(path).get_docs(vocab))
        labels = set(self.cfg.entity_labels)
        missing_ids = 0
        unexpected_labels: Dict[str, int] = {}
        for doc in docs:
            if not doc.user_data.get("note_id") or not doc.user_data.get("patient_id"):
                missing_ids += 1
            for ent in doc.ents:
                if ent.label_ not in labels:
                    unexpected_labels[ent.label_] = unexpected_labels.get(ent.label_, 0) + 1

        if missing_ids:
            raise ValueError(
                f"{path} violates CONTRACT.md: {missing_ids} docs lack note_id or patient_id in doc.user_data"
            )
        if unexpected_labels:
            raise ValueError(
                f"{path} contains labels outside {sorted(labels)}: {unexpected_labels}"
            )

        self.log.info(
            "Loaded %d docs and %d entities from %s",
            len(docs),
            sum(len(doc.ents) for doc in docs),
            path,
        )
        self._validate_attribute_sidecar(path, docs)
        return docs

    def _validate_attribute_sidecar(self, docbin_path: Path, docs: List[Any]) -> None:
        """Read and validate the CONTRACT.md attribute sidecar for a DocBin."""
        sidecar_path = docbin_path.with_name(f"{docbin_path.stem}_attributes.csv")
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"Missing attribute sidecar required by CONTRACT.md: {sidecar_path}"
            )

        required = {
            "note_id",
            "patient_id",
            "start_char",
            "end_char",
            "label",
            "text",
            "assertion",
            "temporality",
            "experiencer",
        }
        gold_keys = {
            (
                str(doc.user_data["note_id"]),
                int(ent.start_char),
                int(ent.end_char),
                ent.label_,
            )
            for doc in docs
            for ent in doc.ents
        }

        with sidecar_path.open(newline="") as f:
            reader = csv.DictReader(f)
            missing_columns = required - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"{sidecar_path} is missing required column(s): {sorted(missing_columns)}"
                )
            sidecar_keys = {
                (
                    str(row["note_id"]),
                    int(row["start_char"]),
                    int(row["end_char"]),
                    row["label"],
                )
                for row in reader
            }

        missing_sidecar = gold_keys - sidecar_keys
        extra_sidecar = sidecar_keys - gold_keys
        if missing_sidecar or extra_sidecar:
            raise ValueError(
                f"{sidecar_path} does not match {docbin_path}: "
                f"missing={len(missing_sidecar)} extra={len(extra_sidecar)}"
            )
        self.log.info("Validated attribute sidecar: %s", sidecar_path)

    def _docs_to_examples(self, nlp, docs: List[Any]) -> List[Any]:
        """Convert gold DocBin docs into spaCy Examples for a specific pipeline."""
        from spacy.training import Example

        examples = []
        for doc in docs:
            pred_doc = nlp.make_doc(doc.text)
            examples.append(Example(pred_doc, doc))
        return examples

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
        import spacy

        self._base_model_source = self.cfg.base_model
        self._loaded_pretrained_base = True
        try:
            nlp = spacy.load(self.cfg.base_model)
            self.log.info("Loaded spaCy base model: %s", self.cfg.base_model)
        except OSError as exc:
            self._base_model_source = 'spacy.blank("en") fallback'
            self._loaded_pretrained_base = False
            self.log.warning(
                "Base model %s is not installed; falling back to spacy.blank('en'). "
                "Mock metrics are only a plumbing check. Original error: %s",
                self.cfg.base_model,
                exc,
            )
            nlp = spacy.blank("en")

        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner", last=True)
            self.log.info("Added missing spaCy 'ner' component")

        return nlp

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
        ner = nlp.get_pipe("ner")
        existing = set(ner.labels)
        for label in self.cfg.entity_labels:
            if label in existing:
                self.log.info("NER label already registered: %s", label)
            else:
                ner.add_label(label)
                self.log.info("Registered NER label: %s", label)

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
        from spacy.util import minibatch

        rng = random.Random(self.cfg.seed)
        rng.shuffle(examples)
        losses: Dict[str, float] = {}
        with nlp.select_pipes(enable=["ner"]):
            for batch in minibatch(examples, size=self.cfg.batch_size):
                nlp.update(batch, sgd=optimizer, drop=drop, losses=losses)
        return {key: float(value) for key, value in losses.items()}

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
        counts: Dict[str, Dict[str, int]] = {
            label: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
            for label in self.cfg.entity_labels
        }

        for example in examples:
            gold_doc = example.reference
            pred_doc = nlp(gold_doc.text)
            gold = {
                (ent.start_char, ent.end_char, ent.label_)
                for ent in gold_doc.ents
                if ent.label_ in counts
            }
            pred = {
                (ent.start_char, ent.end_char, ent.label_)
                for ent in pred_doc.ents
                if ent.label_ in counts
            }
            for _, _, label in gold:
                counts[label]["support"] += 1
            for span in pred & gold:
                counts[span[2]]["tp"] += 1
            for span in pred - gold:
                counts[span[2]]["fp"] += 1
            for span in gold - pred:
                counts[span[2]]["fn"] += 1

        metrics: Dict[str, float] = {}
        total_tp = total_fp = total_fn = total_support = 0
        for label, label_counts in counts.items():
            tp = label_counts["tp"]
            fp = label_counts["fp"]
            fn = label_counts["fn"]
            support = label_counts["support"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics[f"precision_{label}"] = precision
            metrics[f"recall_{label}"] = recall
            metrics[f"f1_{label}"] = f1
            metrics[f"support_{label}"] = float(support)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_support += support

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(total_support),
            }
        )
        return metrics

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
        train_examples = self._docs_to_examples(nlp, train_examples)
        val_examples = self._docs_to_examples(nlp, val_examples)
        if not train_examples:
            raise ValueError("No training examples loaded")
        if not val_examples:
            raise ValueError("No validation examples loaded")

        if getattr(self, "_loaded_pretrained_base", False):
            optimizer = nlp.resume_training()
        else:
            optimizer = nlp.initialize(lambda: train_examples)

        train_losses: List[float] = []
        val_f1_scores: List[float] = []
        eval_epochs: List[int] = []
        best_f1 = -1.0
        best_metrics: Dict[str, float] = {}
        best_checkpoint: Optional[Path] = None
        patience_count = 0
        eval_every = max(1, self.cfg.eval_every_n_epochs)

        for epoch in range(1, self.cfg.n_epochs + 1):
            losses = self.train_spacy_epoch(nlp, optimizer, train_examples, self.cfg.dropout)
            ner_loss = float(losses.get("ner", sum(losses.values()) if losses else 0.0))
            train_losses.append(ner_loss)

            should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == self.cfg.n_epochs
            if should_eval:
                metrics = self.evaluate_spacy(nlp, val_examples)
                val_f1 = float(metrics["f1"])
                val_f1_scores.append(val_f1)
                eval_epochs.append(epoch)
                self.log.info(
                    "Epoch %d/%d: ner_loss=%.4f val_p=%.4f val_r=%.4f val_f1=%.4f",
                    epoch,
                    self.cfg.n_epochs,
                    ner_loss,
                    metrics["precision"],
                    metrics["recall"],
                    val_f1,
                )
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_metrics = metrics
                    best_checkpoint = self.save_checkpoint(nlp, epoch, metrics)
                    patience_count = 0
                else:
                    patience_count += 1

                if patience_count >= self.cfg.early_stopping_patience:
                    self.log.info(
                        "Early stopping after %d validation checks without improvement",
                        patience_count,
                    )
                    break
            else:
                self.log.info(
                    "Epoch %d/%d: ner_loss=%.4f",
                    epoch,
                    self.cfg.n_epochs,
                    ner_loss,
                )

        curve_path = self.plot_training_curves(train_losses, val_f1_scores, eval_epochs)
        self._training_history = {
            "train_losses": train_losses,
            "val_f1_scores": val_f1_scores,
            "eval_epochs": eval_epochs,
            "training_curve_path": str(curve_path),
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        }

        if best_checkpoint is not None:
            import spacy

            nlp = spacy.load(best_checkpoint)
        return nlp, best_metrics

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
        raise NotImplementedError("Track B / HuggingFace training is out of thesis scope")

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
        raise NotImplementedError("Track B / HuggingFace training is out of thesis scope")

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
        checkpoint_dir = self.cfg.model_out_dir / "checkpoints" / f"epoch_{epoch:03d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.track.lower() == "spacy":
            model.to_disk(checkpoint_dir)
        else:
            raise NotImplementedError("Track B / HuggingFace training is out of thesis scope")

        payload = {"epoch": epoch, "metrics": metrics}
        (checkpoint_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        self.log.info("Saved checkpoint: %s", checkpoint_dir)
        return checkpoint_dir

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
        if self.cfg.track.lower() != "spacy":
            raise NotImplementedError("Track B / HuggingFace training is out of thesis scope")

        checkpoints_dir = self.cfg.model_out_dir / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No checkpoints found under {checkpoints_dir}")

        best_path: Optional[Path] = None
        best_f1 = -1.0
        for metrics_path in checkpoints_dir.glob("epoch_*/metrics.json"):
            payload = json.loads(metrics_path.read_text())
            f1 = float(payload.get("metrics", {}).get("f1", -1.0))
            if f1 > best_f1:
                best_f1 = f1
                best_path = metrics_path.parent

        if best_path is None:
            raise FileNotFoundError(f"No checkpoint metrics found under {checkpoints_dir}")

        import spacy

        self.log.info("Loading best checkpoint %s with validation F1 %.4f", best_path, best_f1)
        return spacy.load(best_path)

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
        self.run_dir.mkdir(parents=True, exist_ok=True)
        curve_path = self.run_dir / "training_curves.png"
        csv_path = self.run_dir / "training_curves.csv"
        csv_lines = ["epoch,train_loss,val_f1\n"]
        val_by_epoch = dict(zip(eval_epochs, val_f1_scores))
        for idx, loss in enumerate(train_losses, start=1):
            val = val_by_epoch.get(idx, "")
            csv_lines.append(f"{idx},{loss},{val}\n")
        csv_path.write_text("".join(csv_lines))

        try:
            import matplotlib.pyplot as plt

            fig, ax_loss = plt.subplots(figsize=(7, 4))
            epochs = list(range(1, len(train_losses) + 1))
            ax_loss.plot(epochs, train_losses, marker="o", color="tab:blue", label="NER loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("NER loss", color="tab:blue")
            ax_loss.tick_params(axis="y", labelcolor="tab:blue")

            ax_f1 = ax_loss.twinx()
            ax_f1.plot(eval_epochs, val_f1_scores, marker="s", color="tab:green", label="Val F1")
            ax_f1.set_ylabel("Validation F1", color="tab:green")
            ax_f1.tick_params(axis="y", labelcolor="tab:green")
            ax_f1.set_ylim(0.0, 1.0)

            if val_f1_scores:
                best_epoch = eval_epochs[val_f1_scores.index(max(val_f1_scores))]
                ax_loss.axvline(best_epoch, color="0.5", linestyle="--", linewidth=1)

            fig.tight_layout()
            fig.savefig(curve_path, dpi=150)
            plt.close(fig)
            self.log.info("Saved training curves to %s", curve_path)
            return curve_path
        except Exception as exc:
            self.log.warning("Could not write PNG training curve; CSV saved to %s: %s", csv_path, exc)
            return csv_path

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
        test_docs = self._load_spacy_docbin(self.cfg.test_data_path, model.vocab)
        test_examples = self._docs_to_examples(model, test_docs)
        metrics = self.evaluate_spacy(model, test_examples)

        metrics_path = self.run_dir / "test_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

        rows = ["entity_type,precision,recall,f1,support\n"]
        rows.append(
            f"OVERALL,{metrics['precision']},{metrics['recall']},{metrics['f1']},{metrics['support']}\n"
        )
        for label in self.cfg.entity_labels:
            rows.append(
                f"{label},{metrics[f'precision_{label}']},{metrics[f'recall_{label}']},"
                f"{metrics[f'f1_{label}']},{metrics[f'support_{label}']}\n"
            )
        (self.run_dir / "test_metrics_by_entity.csv").write_text("".join(rows))
        self.log.info("Saved held-out test metrics to %s", metrics_path)
        return metrics

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
        track = self.cfg.track.lower()
        if track != "spacy":
            raise NotImplementedError("Track B / HuggingFace training is out of thesis scope")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        nlp = self.load_spacy_base_model()
        self.add_entity_labels(nlp)
        self._loading_nlp = nlp
        train_data, val_data = self.load_training_data()
        model, val_metrics = self.train_spacy(nlp, train_data, val_data)
        test_metrics = self.evaluate_on_test(model)

        self.cfg.model_out_dir.mkdir(parents=True, exist_ok=True)
        model.to_disk(self.cfg.model_out_dir)
        self.log.info("Saved final spaCy model to %s", self.cfg.model_out_dir)

        run_metrics = {
            "track": track,
            "base_model_requested": self.cfg.base_model,
            "base_model_used": getattr(self, "_base_model_source", self.cfg.base_model),
            "validation": val_metrics,
            "test": test_metrics,
            "history": getattr(self, "_training_history", {}),
        }
        (self.run_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2) + "\n")
        return model
