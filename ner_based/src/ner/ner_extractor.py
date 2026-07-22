# src/ner/ner_extractor.py
"""
NER-based clinical entity extraction pipeline.

Loads a trained spaCy or HuggingFace NER model and runs inference on
preprocessed note chunks to produce a per-note entity DataFrame.

Input  : data/interim/airms/notes_preprocessed/chunk_*.parquet
         models/airms_ner_v1.0/  (trained NER model)
Output : data/interim/airms/ner_extractions/chunk_*.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

LOG = logging.getLogger("mrsa_nlp.ner.extractor")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NERExtractorConfig:
    """
    Parameters for the NER extractor.

    Attributes
    ----------
    preprocessed_notes_dir : Path
        Directory containing preprocessed chunk files.
    out_dir : Path
        Directory for per-note entity extraction results.
    model_path : Path
        Path to the trained NER model directory.
    model_track : str
        "spacy" or "hf" — determines how the model is loaded.
    batch_size : int
        Notes per inference batch.
    apply_negation : bool
        Post-process extracted entities to suppress negated mentions.
    negation_window_tokens : int
        Token look-back window for negation detection.
    save_entity_spans : bool
        Store raw entity span text in the output (useful for QA).
    entity_labels : list of str
        Entity labels to include (filters out any unexpected labels).
    debug : bool
        Limit inference to debug_n_notes rows.
    debug_n_notes : int
    """

    preprocessed_notes_dir: Path = Path("data/interim/airms/notes_preprocessed")
    out_dir: Path = Path("data/interim/airms/ner_extractions")
    model_path: Path = Path("models/airms_ner_v1.0")
    model_track: str = "spacy"
    batch_size: int = 32
    apply_negation: bool = True
    negation_window_tokens: int = 5
    save_entity_spans: bool = False
    entity_labels: List[str] = field(
        default_factory=lambda: ["DISEASE", "MEDICATION", "PROCEDURE"]
    )
    debug: bool = False
    debug_n_notes: int = 100


# ---------------------------------------------------------------------------
# Entity span data class
# ---------------------------------------------------------------------------

@dataclass
class EntitySpan:
    """
    A single extracted entity mention.

    Attributes
    ----------
    text : str
        Surface form of the entity.
    label : str
        Entity type label.
    start : int
        Character start offset in the note.
    end : int
        Character end offset in the note.
    is_negated : bool
        True if the mention is preceded by a negation cue.
    confidence : float
        Model confidence score (if available; else 1.0).
    """

    text: str
    label: str
    start: int
    end: int
    is_negated: bool = False
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class NERExtractor:
    """
    Runs a trained NER model on preprocessed clinical note chunks.

    Parameters
    ----------
    config : NERExtractorConfig
    logger : logging.Logger, optional

    Example
    -------
    >>> from src.ner.ner_extractor import NERExtractorConfig, NERExtractor
    >>> cfg = NERExtractorConfig(debug=True, debug_n_notes=50)
    >>> extractor = NERExtractor(cfg)
    >>> extractor.load_model()
    >>> extractor.run()
    """

    def __init__(
        self,
        config: NERExtractorConfig,
        logger: logging.Logger = LOG,
    ) -> None:
        self.cfg = config
        self.log = logger
        self._model: Any = None
        self._tokenizer: Any = None  # HF only

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the trained NER model from cfg.model_path into self._model.

        Track A (spaCy):
            ``self._model = spacy.load(cfg.model_path)``
        Track B (HuggingFace):
            Load tokenizer into self._tokenizer and model into self._model.

        Raises
        ------
        FileNotFoundError
            If cfg.model_path does not exist. Prompts user to run the
            training pipeline first.
        """
        if not self.cfg.model_path.exists():
            raise FileNotFoundError(
                f"NER model path does not exist: {self.cfg.model_path}. Run training first."
            )

        if self.cfg.model_track.lower() == "spacy":
            import spacy

            self._model = spacy.load(self.cfg.model_path)
            self.log.info("Loaded spaCy NER model from %s", self.cfg.model_path)
            return

        if self.cfg.model_track.lower() == "hf":
            raise NotImplementedError("Track B / HuggingFace extraction is out of thesis scope")

        raise ValueError(f"Unknown model_track: {self.cfg.model_track}")

    # ------------------------------------------------------------------
    # Single-note inference
    # ------------------------------------------------------------------

    def extract_entities_spacy(self, text: str) -> List[EntitySpan]:
        """
        Run spaCy NER on a single note text.

        Parameters
        ----------
        text : str
            Preprocessed note text.

        Returns
        -------
        list of EntitySpan
            All entity mentions found by the model.

        Notes
        -----
        - ``doc = self._model(text)``
        - For each ``ent`` in ``doc.ents``:
            - Filter by cfg.entity_labels.
            - Create an EntitySpan with text, label_, start_char, end_char.
        - confidence is not available from spaCy NER; default to 1.0.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded; call load_model() first")
        doc = self._model(text)
        entities = []
        labels = set(self.cfg.entity_labels)
        for ent in doc.ents:
            if ent.label_ not in labels:
                continue
            entities.append(
                EntitySpan(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,
                )
            )
        return entities

    def extract_entities_hf(self, text: str) -> List[EntitySpan]:
        """
        Run HuggingFace token-classification NER on a single note text.

        Parameters
        ----------
        text : str

        Returns
        -------
        list of EntitySpan

        Notes
        -----
        - Tokenize with self._tokenizer (truncation, padding).
        - Run model forward pass; decode BIO predictions to entity spans.
        - Use ``transformers.TokenClassificationPipeline`` if simpler, or
          implement manual BIO decoding for more control.
        - Map subword offsets back to character offsets in the original text.
        """
        raise NotImplementedError("Track B / HuggingFace extraction is out of thesis scope")

    def detect_negation(
        self,
        text: str,
        entities: List[EntitySpan],
    ) -> List[EntitySpan]:
        """
        Mark negated entity mentions using a window-based heuristic.

        Parameters
        ----------
        text : str
            Full note text.
        entities : list of EntitySpan

        Returns
        -------
        list of EntitySpan
            Same list with is_negated flags set where appropriate.

        Notes
        -----
        - Reuse the negation cue list from the rule-based NegationHandler.
        - For each entity, check whether a negation cue appears within
          cfg.negation_window_tokens tokens before the entity start.
        - Do NOT remove negated entities — keep them with is_negated=True
          so downstream aggregation can choose to include or exclude them.
        """
        raise NotImplementedError("blocked: needs index-event definition from supervisor")

    def extract_from_note(
        self,
        text: str,
        note_id: Optional[str] = None,
    ) -> List[EntitySpan]:
        """
        Extract entities from a single note with optional negation marking.

        Parameters
        ----------
        text : str
        note_id : str, optional
            Used for debug logging.

        Returns
        -------
        list of EntitySpan

        Notes
        -----
        - Dispatch to extract_entities_spacy() or extract_entities_hf()
          based on cfg.model_track.
        - If cfg.apply_negation: detect_negation().
        """
        if self.cfg.model_track.lower() == "spacy":
            entities = self.extract_entities_spacy(text)
        elif self.cfg.model_track.lower() == "hf":
            entities = self.extract_entities_hf(text)
        else:
            raise ValueError(f"Unknown model_track: {self.cfg.model_track}")

        if self.cfg.apply_negation:
            # ConText/assertion/temporality behavior is intentionally blocked.
            return self.detect_negation(text, entities)
        return entities

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def entities_to_features(
        self,
        entities: List[EntitySpan],
    ) -> Dict[str, int]:
        """
        Convert a list of EntitySpan objects to a feature dict.

        Features produced:
          has_{entity_type} (binary, 1 if any non-negated entity of that type)
          count_{entity_type} (total non-negated entities of that type)
          has_{entity_type}_negated (binary, 1 if any negated entity)

        Parameters
        ----------
        entities : list of EntitySpan

        Returns
        -------
        dict of {str: int}

        Notes
        -----
        - Count only entities whose label is in cfg.entity_labels.
        - Negated entities contribute to has_*_negated but NOT to has_* / count_*.
        """
        features: Dict[str, int] = {}
        for label in self.cfg.entity_labels:
            features[f"has_{label}"] = 0
            features[f"count_{label}"] = 0
            features[f"has_{label}_negated"] = 0

        for ent in entities:
            if ent.label not in self.cfg.entity_labels:
                continue
            if ent.is_negated:
                features[f"has_{ent.label}_negated"] = 1
            else:
                features[f"has_{ent.label}"] = 1
                features[f"count_{ent.label}"] += 1
        return features

    def extract_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run NER extraction on every row in a notes chunk DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed notes with NOTE_TEXT_CLEAN column.

        Returns
        -------
        pd.DataFrame
            Input columns + has_*, count_*, has_*_negated feature columns,
            and optionally ENTITY_SPANS (JSON list of span dicts if
            cfg.save_entity_spans is True).

        Notes
        -----
        - Use tqdm for a progress bar.
        - For large DataFrames, batch the texts and run spaCy's
          ``nlp.pipe(texts, batch_size=cfg.batch_size)`` for efficiency.
        """
        rows = []
        text_col = "NOTE_TEXT_CLEAN" if "NOTE_TEXT_CLEAN" in df.columns else "note_text"
        if text_col not in df.columns:
            raise ValueError("Expected NOTE_TEXT_CLEAN or note_text column in preprocessed notes")

        for _, row in df.iterrows():
            entities = self.extract_from_note(str(row[text_col]), note_id=str(row.get("NOTE_ID", "")))
            features = self.entities_to_features(entities)
            out_row = row.to_dict()
            out_row.update(features)
            if self.cfg.save_entity_spans:
                import json

                out_row["ENTITY_SPANS"] = json.dumps([ent.__dict__ for ent in entities])
            rows.append(out_row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def list_preprocessed_chunks(self) -> List[Path]:
        """
        Return sorted list of preprocessed note chunk files.

        Returns
        -------
        list of Path

        Raises
        ------
        FileNotFoundError
            If cfg.preprocessed_notes_dir is missing or empty.
        """
        if not self.cfg.preprocessed_notes_dir.exists():
            raise FileNotFoundError(f"Missing preprocessed notes dir: {self.cfg.preprocessed_notes_dir}")
        chunks = sorted(self.cfg.preprocessed_notes_dir.glob("*.parquet"))
        if not chunks:
            raise FileNotFoundError(f"No preprocessed Parquet chunks found in {self.cfg.preprocessed_notes_dir}")
        return chunks

    def run(self) -> None:
        """
        Run NER extraction on all preprocessed note chunks (resume-safe).

        Steps
        -----
        1. load_model() if not already loaded.
        2. list_preprocessed_chunks().
        3. For each chunk:
            a. Skip if output chunk exists.
            b. Load chunk.
            c. extract_batch().
            d. Write to cfg.out_dir/chunk_NNNN.parquet.
            e. Log per-chunk stats.
        4. Log overall summary (entity counts by type).

        Raises
        ------
        FileNotFoundError
            If no preprocessed chunks found.
        RuntimeError
            If model has not been loaded (call load_model() first).
        """
        if self._model is None:
            self.load_model()

        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        for chunk_path in self.list_preprocessed_chunks():
            out_path = self.cfg.out_dir / chunk_path.name
            if out_path.exists():
                self.log.info("Skipping existing extraction chunk: %s", out_path)
                continue
            df = pd.read_parquet(chunk_path)
            if self.cfg.debug:
                df = df.head(self.cfg.debug_n_notes)
            out_df = self.extract_batch(df)
            out_df.to_parquet(out_path, index=False)
            self.log.info("Wrote extraction chunk %s with shape %s", out_path, out_df.shape)
