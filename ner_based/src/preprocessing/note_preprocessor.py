# src/preprocessing/note_preprocessor.py
"""
Preprocessing pipeline for raw clinical notes — NER-based project.

Prepares note text for Named Entity Recognition by applying clinical-text
normalisation (whitespace, abbreviation expansion, optional section splitting).
Unlike the rule-based project, section segmentation here is more important
because NER performance can improve when entities are extracted
section-specifically (e.g. medications section only for drug NER).

Input  : data/interim/airms/notes/chunk_*.parquet
Output : data/interim/airms/notes_preprocessed/chunk_*.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LOG = logging.getLogger("mrsa_nlp.ner.preprocess")

# Common clinical abbreviations to expand before NER
CLINICAL_ABBREVIATIONS: Dict[str, str] = {
    r"\bs/p\b":     "status post",
    r"\bw/o\b":     "without",
    r"\bw/\b":      "with",
    r"\bc/o\b":     "complains of",
    r"\bSOB\b":     "shortness of breath",
    r"\bDOE\b":     "dyspnea on exertion",
    r"\bHTN\b":     "hypertension",
    r"\bDM\b":      "diabetes mellitus",
    r"\bDM2\b":     "type 2 diabetes mellitus",
    r"\bCKD\b":     "chronic kidney disease",
    r"\bESRD\b":    "end-stage renal disease",
    r"\bICU\b":     "intensive care unit",
    r"\bSNF\b":     "skilled nursing facility",
    r"\bRA\b":      "rheumatoid arthritis",
    r"\bSLE\b":     "systemic lupus erythematosus",
    r"\bHIV\b":     "human immunodeficiency virus",
    r"\bAIDS\b":    "acquired immunodeficiency syndrome",
    r"\bTx\b":      "transplant",
    r"\bCVL\b":     "central venous line",
    r"\bCVC\b":     "central venous catheter",
    r"\bPICC\b":    "peripherally inserted central catheter",
    r"\bHD\b":      "hemodialysis",
    r"\bAbx\b":     "antibiotics",
    r"\bVanc\b":    "vancomycin",
    r"\bPred\b":    "prednisone",
    r"\bMTX\b":     "methotrexate",
    r"\bMMF\b":     "mycophenolate mofetil",
    r"\bIVDA\b":    "intravenous drug abuse",
    r"\bIVDU\b":    "intravenous drug use",
}

# Section header patterns (regex)
SECTION_HEADERS: Dict[str, str] = {
    "HPI":          r"(?:chief complaint|history of present illness|hpi)\s*[:]\s*",
    "PMH":          r"(?:past medical history|pmh)\s*[:]\s*",
    "MEDICATIONS":  r"(?:medications|current medications|meds)\s*[:]\s*",
    "ALLERGIES":    r"(?:allergies|allergy)\s*[:]\s*",
    "ASSESSMENT":   r"(?:assessment|impression)\s*[:]\s*",
    "PLAN":         r"(?:plan|treatment plan)\s*[:]\s*",
    "SOCIAL":       r"(?:social history|sh)\s*[:]\s*",
    "FAMILY":       r"(?:family history|fh)\s*[:]\s*",
    "REVIEW":       r"(?:review of systems|ros)\s*[:]\s*",
    "PHYSICAL":     r"(?:physical exam|physical examination|pe)\s*[:]\s*",
    "DISCHARGE":    r"(?:discharge condition|discharge instructions|discharge summary)\s*[:]\s*",
}


@dataclass
class NERPreprocessorConfig:
    """
    Tunable parameters for the NER note preprocessor.

    Attributes
    ----------
    raw_notes_dir : Path
        Directory containing chunk_*.parquet files from the cohort builder.
    out_dir : Path
        Directory where pre-processed chunk files are written.
    min_note_length : int
        Minimum character length to keep.
    max_note_length : int
        Maximum character length; longer notes are truncated.
    lowercase : bool
        Lower-case the text.  NER models may handle case internally — set to
        False for BERT-based models that are case-sensitive.
    remove_extra_whitespace : bool
        Normalise multi-space to single space.
    expand_abbreviations : bool
        Replace clinical shorthands using CLINICAL_ABBREVIATIONS map.
    segment_sections : bool
        Identify and label note sections.
    keep_original_text : bool
        Retain the raw NOTE_TEXT column alongside NOTE_TEXT_CLEAN.
    max_tokens_per_note : int
        Approximate token ceiling (whitespace-split); notes exceeding this are
        split into overlapping sub-notes for BERT processing.  0 = no split.
    overlap_tokens : int
        Overlap between consecutive sub-note windows when splitting long notes.
    debug : bool
        Limit processing to debug_n_notes rows.
    debug_n_notes : int
        Row limit in debug mode.
    """

    raw_notes_dir: Path = Path("data/interim/airms/notes")
    out_dir: Path = Path("data/interim/airms/notes_preprocessed")
    min_note_length: int = 50
    max_note_length: int = 100_000
    lowercase: bool = False          # keep case for transformer models
    remove_extra_whitespace: bool = True
    expand_abbreviations: bool = True
    segment_sections: bool = True
    keep_original_text: bool = False
    max_tokens_per_note: int = 0     # 0 = disabled; set 500 for BERT window
    overlap_tokens: int = 50
    debug: bool = False
    debug_n_notes: int = 100


class NERNotePreprocessor:
    """
    Loads raw note chunks and applies NER-oriented text normalisation.

    The key difference from the rule-based preprocessor is:
    - Case is preserved by default (transformer models are case-sensitive).
    - Long notes can be windowed for BERT's 512-token limit.
    - Section segmentation is enabled by default.

    Parameters
    ----------
    config : NERPreprocessorConfig
    logger : logging.Logger, optional

    Example
    -------
    >>> from src.preprocessing.note_preprocessor import NERPreprocessorConfig, NERNotePreprocessor
    >>> cfg = NERPreprocessorConfig(debug=True)
    >>> pp = NERNotePreprocessor(cfg)
    >>> pp.run()
    """

    def __init__(
        self,
        config: NERPreprocessorConfig,
        logger: logging.Logger = LOG,
    ) -> None:
        self.cfg = config
        self.log = logger

    def list_chunk_files(self) -> List[Path]:
        """
        Return sorted list of raw note chunk Parquet files.

        Returns
        -------
        list of Path

        Raises
        ------
        FileNotFoundError
            If cfg.raw_notes_dir is missing or has no chunk files.
            Prompt user to run the cohort builder first.
        """
        pass

    def load_chunk(self, chunk_path: Path) -> pd.DataFrame:
        """
        Load a single raw note chunk.

        Parameters
        ----------
        chunk_path : Path

        Returns
        -------
        pd.DataFrame
            Columns: NOTE_ID, PERSON_ID, NOTE_DATE, NOTE_TEXT,
            NOTE_TYPE_CONCEPT_ID, VISIT_OCCURRENCE_ID.
        """
        pass

    def clean_whitespace(self, text: str) -> str:
        """
        Collapse multiple whitespace to single space; strip edges.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        pass

    def expand_abbreviations(self, text: str) -> str:
        """
        Replace clinical abbreviations with full terms (regex, case-insensitive).

        Parameters
        ----------
        text : str
            Note text.

        Returns
        -------
        str
            Text with abbreviations replaced.

        Notes
        -----
        Uses CLINICAL_ABBREVIATIONS dict defined at module level.
        """
        pass

    def segment_sections(self, text: str) -> Dict[str, str]:
        """
        Split a note into labelled clinical sections.

        Parameters
        ----------
        text : str
            Cleaned note text.

        Returns
        -------
        dict of {str: str}
            Keys from SECTION_HEADERS + "FULL_TEXT" and "OTHER".
            Values are section text strings.

        Notes
        -----
        - Uses SECTION_HEADERS regex patterns to find boundaries.
        - Text between recognised headers is assigned to the preceding
          section key.
        - Unassigned leading text is labelled "OTHER".
        """
        pass

    def window_long_note(self, text: str) -> List[str]:
        """
        Split a long note into overlapping token windows for BERT.

        Parameters
        ----------
        text : str
            Cleaned note text.

        Returns
        -------
        list of str
            If the note is within the cfg.max_tokens_per_note budget: [text].
            Otherwise: list of windowed sub-notes, each ≤ max_tokens_per_note
            tokens with cfg.overlap_tokens overlap between consecutive windows.

        Notes
        -----
        - Only active when cfg.max_tokens_per_note > 0.
        - Tokenise by whitespace for the windowing logic (not subword tokens).
        - Sub-notes are joined on their original whitespace.
        """
        pass

    def process_note_text(self, text: str) -> str:
        """
        Apply the full cleaning sequence to a single note string.

        Order:
        1. clean_whitespace()
        2. expand_abbreviations()  if cfg.expand_abbreviations
        3. lowercase()             if cfg.lowercase

        Parameters
        ----------
        text : str

        Returns
        -------
        str
            Cleaned text, or empty string if input is None/NaN.
        """
        pass

    def filter_notes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove null, too-short, or too-long notes.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        - Drop null/empty NOTE_TEXT.
        - Drop notes shorter than cfg.min_note_length characters.
        - Truncate to cfg.max_note_length characters.
        - Log dropped counts.
        """
        pass

    def deduplicate_notes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop exact duplicate notes (same PERSON_ID + NOTE_TEXT).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        pass

    def process_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full preprocessing to a chunk DataFrame.

        Steps
        -----
        1. filter_notes()
        2. Apply process_note_text() → NOTE_TEXT_CLEAN
        3. Optionally segment_sections() → NOTE_SECTIONS (JSON string)
        4. Optionally window_long_note() → expand rows if cfg.max_tokens_per_note > 0
        5. deduplicate_notes()

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        pass

    def run(self) -> None:
        """
        Execute preprocessing on all raw note chunks (resume-safe).

        Raises
        ------
        FileNotFoundError
            If no raw chunks are found.
        """
        pass
