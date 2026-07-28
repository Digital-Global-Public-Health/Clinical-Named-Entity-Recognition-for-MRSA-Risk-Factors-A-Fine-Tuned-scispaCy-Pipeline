# src/cli.py
"""
Command-line interface for the MRSA NLP NER-based pipeline.

Usage (from project root, with conda env activated):

    python -m src.cli --help

    # Step 1 — build cohort + mine notes
    python -m src.cli build-cohort [OPTIONS]

    # Step 2 — preprocess raw notes
    python -m src.cli preprocess [OPTIONS]

    # Step 3 — annotate a sample for NER training (exports guidelines)
    python -m src.cli prepare-annotations [OPTIONS]

    # Extract one patient's notes for pre-annotation
    python -m src.cli extract-patient [OPTIONS]

    # Step 4 — train / fine-tune the NER model
    python -m src.cli train [OPTIONS]

    # Step 5 — run NER extraction on all notes
    python -m src.cli extract [OPTIONS]

    # Step 6 — aggregate NER features to visit level
    python -m src.cli aggregate-features [OPTIONS]

    # Step 7 — evaluate extraction quality
    python -m src.cli evaluate [OPTIONS]

    # Run the full pipeline end-to-end
    python -m src.cli run-pipeline [OPTIONS]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print

from src.utils_logging import configure_logging, logger, make_run_dir, save_config_snapshot, log_timing
from src.utils_db import connect_hana
from src.utils_seed import set_seed, GLOBAL_SEED
from src.cohort.cohort_builder import CohortConfig, CohortBuilder
from src.preprocessing.note_preprocessor import NERPreprocessorConfig, NERNotePreprocessor
from src.ner.annotation_schema import AnnotationSchemaConfig, AnnotationSchema
from src.ner.model_trainer import NERTrainerConfig, NERModelTrainer
from src.ner.mock_data import MockNERDataConfig, generate_mock_ner_data
from src.ner.ner_extractor import NERExtractorConfig, NERExtractor
from src.ner.extract_patient import (
    DEFAULT_NOTE_TITLES,
    DEFAULT_OUTPUT_PATH,
    LOG as extract_patient_logger,
    PatientExtractionError,
    PatientNoteExtractionConfig,
    extract_patient_notes,
    parse_note_titles,
)
from src.ner.preannotate import (
    OllamaClient,
    OllamaConfig,
    load_notes,
    run_preannotation,
    synthetic_canned_responses,
    synthetic_fixture_notes,
)
from src.ner.preannotation_serializers import write_contract_docbin, write_webanno_tsv3
from src.features.feature_aggregator import NERAggregatorConfig, NERFeatureAggregator
from src.evaluation.evaluator import NEREvaluatorConfig, NEREvaluator

app = typer.Typer(
    add_completion=False,
    help="MRSA NLP — NER-based clinical note extraction and model training pipeline.",
)


# ---------------------------------------------------------------------------
# Global callback: configure logging and seeding
# ---------------------------------------------------------------------------

@app.callback()
def _configure(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level: DEBUG | INFO | WARNING | ERROR"
    ),
    seed: int = typer.Option(
        GLOBAL_SEED, "--seed", help=f"Random seed for reproducibility (default: {GLOBAL_SEED})"
    ),
) -> None:
    """Global CLI options, logging setup, and seed initialization."""
    run_name = ctx.invoked_subcommand or "cli"
    if run_name == "extract-patient":
        numeric = getattr(logging, log_level.upper(), logging.INFO)
        previous_level = extract_patient_logger.level
        previous_propagate = extract_patient_logger.propagate
        handler = logging.StreamHandler()
        handler.setLevel(numeric)
        handler.setFormatter(logging.Formatter("%(message)s"))
        extract_patient_logger.setLevel(numeric)
        extract_patient_logger.propagate = False
        extract_patient_logger.addHandler(handler)

        def _close_extract_patient_logging() -> None:
            extract_patient_logger.removeHandler(handler)
            handler.close()
            extract_patient_logger.setLevel(previous_level)
            extract_patient_logger.propagate = previous_propagate

        ctx.call_on_close(_close_extract_patient_logging)
        return

    run_dir = configure_logging(log_level, run_name=run_name)
    set_seed(seed)
    logger.info("Log level : %s", log_level.upper())
    logger.info("Seed      : %d", seed)
    logger.info("Run dir   : %s", run_dir)


# ---------------------------------------------------------------------------
# 1. build-cohort
# ---------------------------------------------------------------------------

@app.command(help="Load the MRSA cohort from mrsa_risk_predictions and mine notes from CDMPHI.NOTES.")
@log_timing
def build_cohort(
    schema: str = typer.Option("CDMPHI", help="HANA schema name."),
    chunk_size: int = typer.Option(500, help="Persons per note-mining chunk."),
    min_note_date: str = typer.Option("2014-07-14", help="Earliest note date (YYYY-MM-DD)."),
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Debug mode: limit to a small sample."),
    debug_n_persons: int = typer.Option(20, help="Persons to process in debug mode."),
    seed: int = typer.Option(GLOBAL_SEED, "--seed", help=f"Random seed (passed to mrsa_risk_predictions cohort loader; default: {GLOBAL_SEED})."),
) -> None:
    """
    Pipeline Step 1 — Build cohort and mine clinical notes.

    Reads the matched-pairs cohort from mrsa_risk_predictions, resolves MRNs,
    saves mrsa_cohort_person_list.parquet, then fetches notes from CDMPHI.NOTES
    in person-level chunks (resume-safe).

    The seed is used in the underlying cohort builder for reproducible control sampling.
    """
    cfg = CohortConfig(
        schema=schema,
        chunk_size=chunk_size,
        min_note_date=min_note_date,
        debug=debug,
        debug_n_persons=debug_n_persons,
    )

    save_config_snapshot(
        cfg.__dict__ | {"pipeline_step": "build_cohort"},
        run_dir=_current_run_dir(),
    )

    conn = connect_hana()
    builder = CohortBuilder(cfg, conn)
    person_df = builder.run()

    logger.info(
        "Cohort built: %d persons  (%d cases, %d controls)",
        len(person_df) if person_df is not None else 0,
        (person_df["LABEL"] == 1).sum() if person_df is not None else 0,
        (person_df["LABEL"] == 0).sum() if person_df is not None else 0,
    )


# ---------------------------------------------------------------------------
# 2. preprocess
# ---------------------------------------------------------------------------

@app.command(help="Clean and normalise raw clinical note chunks for NER inference.")
@log_timing
def preprocess(
    raw_notes_dir: Path = typer.Option(
        Path("data/interim/airms/notes"),
        help="Directory of raw note chunk Parquet files.",
    ),
    out_dir: Path = typer.Option(
        Path("data/interim/airms/notes_preprocessed"),
        help="Directory for preprocessed note chunks.",
    ),
    max_tokens: int = typer.Option(
        0, help="Max tokens per note for BERT windowing (0 = no windowing)."
    ),
    expand_abbrev: bool = typer.Option(True, "--expand-abbrev/--no-expand-abbrev"),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
    debug_n_notes: int = typer.Option(200),
) -> None:
    """
    Pipeline Step 2 — Preprocess raw note chunks.

    Applies whitespace normalisation and abbreviation expansion.  Optionally
    windows long notes for BERT 512-token limit.  Skips already-processed
    chunks (resume-safe).

    Note: lowercase is disabled by default to preserve case for BERT models.
    """
    cfg = NERPreprocessorConfig(
        raw_notes_dir=raw_notes_dir,
        out_dir=out_dir,
        max_tokens_per_note=max_tokens,
        expand_abbreviations=expand_abbrev,
        debug=debug,
        debug_n_notes=debug_n_notes,
    )

    save_config_snapshot(
        cfg.__dict__ | {"pipeline_step": "preprocess"},
        run_dir=_current_run_dir(),
    )

    pp = NERNotePreprocessor(cfg)
    pp.run()


# ---------------------------------------------------------------------------
# 3. prepare-annotations
# ---------------------------------------------------------------------------

@app.command(help="Export annotation schema and guidelines for NER training data creation.")
@log_timing
def prepare_annotations(
    entity_types: str = typer.Option(
        "DISEASE,MEDICATION,PROCEDURE",
        help="Comma-separated entity types to include.",
    ),
    include_severity: bool = typer.Option(
        False, "--include-severity", help="Add optional SEVERITY entity type."
    ),
    guidelines_out: Path = typer.Option(
        Path("annotations/annotation_guidelines.md"),
        help="Where to write the Markdown guidelines document.",
    ),
    schema_json_out: Optional[Path] = typer.Option(
        None, help="Where to write schema.json for annotation tools (optional)."
    ),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
) -> None:
    """
    Pipeline Step 3 — Prepare annotation resources.

    Exports the entity annotation schema to a Markdown guidelines document
    and optionally a JSON schema file for use with labelling tools (e.g. Label
    Studio, Prodigy).  Human annotators should read the guidelines before
    creating training data.
    """
    labels = [t.strip() for t in entity_types.split(",")]
    if include_severity and "SEVERITY" not in labels:
        labels.append("SEVERITY")

    cfg = AnnotationSchemaConfig(
        entity_types=labels,
        guidelines_out_path=guidelines_out,
        debug=debug,
    )

    save_config_snapshot(
        {"entity_types": labels, "guidelines_out": str(guidelines_out), "pipeline_step": "prepare_annotations"},
        run_dir=_current_run_dir(),
    )

    schema = AnnotationSchema(cfg)
    guidelines_path = schema.export_guidelines()
    logger.info("Guidelines written to: %s", guidelines_path)

    if schema_json_out:
        json_path = schema.export_schema_json(schema_json_out)
        logger.info("Schema JSON written to: %s", json_path)

    logger.info("Configured entity types: %s", schema.get_entity_labels())


# ---------------------------------------------------------------------------
# 3b. extract-patient
# ---------------------------------------------------------------------------

@app.command(
    "extract-patient",
    help="Extract one cohort patient's full notes for NER pre-annotation.",
)
def extract_patient(
    notes_parquet: Path = typer.Option(
        ...,
        "--notes-parquet",
        help="Cohort notes parquet containing PERSON_ID and clinical note columns.",
    ),
    cohort_csv: Path = typer.Option(
        ...,
        "--cohort-csv",
        help="Cohort CSV containing PERSON_ID and LABEL.",
    ),
    person_id: str = typer.Option(
        ...,
        "--person-id",
        help="Single PERSON_ID to extract; interpreted using the parquet PERSON_ID type.",
    ),
    note_titles: str = typer.Option(
        ",".join(DEFAULT_NOTE_TITLES),
        "--note-titles",
        help="Comma-separated NOTE_TITLE values to retain.",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--output",
        help="Output parquet path. Existing files are replaced; directories are never removed.",
    ),
) -> None:
    """Extract one patient's full notes with a read-time parquet filter."""
    try:
        titles = parse_note_titles(note_titles)
        extract_patient_notes(
            PatientNoteExtractionConfig(
                notes_parquet=notes_parquet,
                cohort_csv=cohort_csv,
                person_ids=(person_id,),
                note_titles=titles,
                output=output,
                require_cohort_membership=True,
            )
        )
    except PatientExtractionError as exc:
        raise typer.BadParameter(str(exc)) from exc


# ---------------------------------------------------------------------------
# 3c. preannotate
# ---------------------------------------------------------------------------

@app.command(help="Ask Ollama for NER pre-annotations, verify spans, and export review artifacts.")
@log_timing
def preannotate(
    input_path: Optional[Path] = typer.Option(
        None,
        "--input-path",
        help="Note input file or directory: JSON, JSONL, TXT, or parquet. Not used with --synthetic-fixtures.",
    ),
    out_dir: Path = typer.Option(
        Path("annotations/preannotations"),
        help="Output directory for verified JSON, INCEpTION TSV, and CONTRACT.md artifacts.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Ollama model name. Defaults to OLLAMA_MODEL from .env.",
    ),
    synthetic_fixtures: bool = typer.Option(
        False,
        "--synthetic-fixtures",
        help="Run offline on synthetic notes with canned model responses; no Ollama server required.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing per-note JSON artifacts. Default is resume-safe skip.",
    ),
    write_inception: bool = typer.Option(
        True,
        "--write-inception/--no-write-inception",
        help="Write INCEpTION-importable WebAnno TSV 3.3 files from verified JSON.",
    ),
    write_contract: bool = typer.Option(
        True,
        "--write-contract/--no-write-contract",
        help="Write CONTRACT.md DocBin plus empty reserved sidecar from verified JSON.",
    ),
) -> None:
    """
    Pipeline Step 3b - LLM pre-annotation draft.

    The LLM is only an assistant. Its strings are verified against the source
    note before offsets are written. Gold annotation is spans and labels only.

    Offline WSL smoke test:

        python -m src.cli preannotate --synthetic-fixtures --out-dir annotations/preannotations/synthetic --overwrite

    Real Minerva run:

        python -m src.cli preannotate --input-path data/interim/airms/notes_preprocessed --out-dir annotations/preannotations/minerva --model "$OLLAMA_MODEL"
    """
    if synthetic_fixtures:
        notes = synthetic_fixture_notes()
        canned = synthetic_canned_responses()
        client = None
    else:
        if input_path is None:
            raise typer.BadParameter("--input-path is required unless --synthetic-fixtures is used")
        notes = load_notes(input_path)
        canned = None
        client = OllamaClient(OllamaConfig.from_env(model=model))

    save_config_snapshot(
        {
            "pipeline_step": "preannotate",
            "input_path": input_path,
            "out_dir": out_dir,
            "model": model,
            "synthetic_fixtures": synthetic_fixtures,
            "overwrite": overwrite,
            "write_inception": write_inception,
            "write_contract": write_contract,
        },
        run_dir=_current_run_dir(),
    )

    totals = run_preannotation(
        notes=notes,
        out_dir=out_dir,
        client=client,
        canned_responses=canned,
        overwrite=overwrite,
    )
    print(totals.format_block("Run total"))

    if write_inception:
        inception_summary = write_webanno_tsv3(out_dir / "verified", out_dir / "inception_webanno_tsv3")
        print(inception_summary.format_block("WebAnno TSV stats"))
    if write_contract:
        write_contract_docbin(out_dir / "verified", out_dir / "contract_docbin")


# ---------------------------------------------------------------------------
# 4. train
# ---------------------------------------------------------------------------

@app.command(help="Train or fine-tune a NER model on annotated clinical notes.")
@log_timing
def train(
    track: str = typer.Option(
        "spacy",
        help="Training track: 'spacy' (scispaCy fine-tuning) or 'hf' (BioClinicalBERT).",
    ),
    base_model: str = typer.Option(
        "en_core_sci_sm",
        help="Base model: spaCy — 'en_core_sci_sm'/'en_core_sci_lg'; "
             "HF — 'emilyalsentzer/Bio_ClinicalBERT'.",
    ),
    train_data: Path = typer.Option(
        Path("annotations/train.spacy"),
        help="Training annotations (.spacy DocBin or CSV).",
    ),
    val_data: Path = typer.Option(
        Path("annotations/val.spacy"),
        help="Validation annotations.",
    ),
    test_data: Path = typer.Option(
        Path("annotations/test.spacy"),
        help="Held-out test annotations (evaluated only at end).",
    ),
    model_out_dir: Path = typer.Option(
        Path("models/airms_ner_v1.0"),
        help="Directory to save model checkpoints and final model.",
    ),
    n_epochs: int = typer.Option(30, help="Number of training epochs."),
    batch_size: int = typer.Option(16, help="Mini-batch size."),
    dropout: float = typer.Option(0.3, help="Dropout rate."),
    learning_rate: float = typer.Option(1e-3, help="Initial learning rate."),
    device: str = typer.Option("cpu", help="Device: 'cpu', 'gpu', or 'cuda:0'."),
    seed: int = typer.Option(GLOBAL_SEED, help=f"Random seed for reproducibility (default: {GLOBAL_SEED})."),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
    debug_n_examples: int = typer.Option(50, help="Max training examples in debug mode."),
) -> None:
    """
    Pipeline Step 4 — NER model training / fine-tuning.

    Track A (spaCy): Fine-tunes en_core_sci_sm on annotated clinical notes
    using spaCy's training loop with early stopping.

    Track B (HF): Fine-tunes Bio_ClinicalBERT with a token-classification head
    using HuggingFace Transformers + seqeval metrics.  Requires GPU on Minerva.
    """
    _, run_dir = make_run_dir(f"train_{track}")

    cfg = NERTrainerConfig(
        track=track,
        base_model=base_model,
        train_data_path=train_data,
        val_data_path=val_data,
        test_data_path=test_data,
        model_out_dir=model_out_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        dropout=dropout,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        debug=debug,
        debug_n_examples=debug_n_examples,
    )

    save_config_snapshot(cfg.__dict__ | {"pipeline_step": "train"}, run_dir)

    schema = AnnotationSchema(AnnotationSchemaConfig())
    trainer = NERModelTrainer(cfg, schema, run_dir)
    model = trainer.run()

    logger.info("Training complete.  Model saved to: %s", model_out_dir)


# ---------------------------------------------------------------------------
# 5. extract
# ---------------------------------------------------------------------------

@app.command(help="Run NER inference on preprocessed note chunks.")
@log_timing
def extract(
    preprocessed_dir: Path = typer.Option(
        Path("data/interim/airms/notes_preprocessed"),
        help="Directory of preprocessed note chunk Parquet files.",
    ),
    out_dir: Path = typer.Option(
        Path("data/interim/airms/ner_extractions"),
        help="Directory for per-note NER extraction results.",
    ),
    model_path: Path = typer.Option(
        Path("models/airms_ner_v1.0"),
        help="Path to the trained NER model directory.",
    ),
    model_track: str = typer.Option(
        "spacy", help="Model type: 'spacy' or 'hf'."
    ),
    batch_size: int = typer.Option(32, help="Notes per inference batch."),
    no_negation: bool = typer.Option(
        False, "--no-negation", help="Disable negation detection."
    ),
    negation_window: int = typer.Option(5, help="Negation look-back window (tokens)."),
    save_spans: bool = typer.Option(
        False, "--save-spans", help="Store raw entity span text in output."
    ),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
    debug_n_notes: int = typer.Option(100),
) -> None:
    """
    Pipeline Step 5 — NER extraction.

    Loads the trained NER model and runs batch inference on all preprocessed
    note chunks.  Skips already-processed chunks (resume-safe).  Detected
    entities are flagged for negation before saving.
    """
    cfg = NERExtractorConfig(
        preprocessed_notes_dir=preprocessed_dir,
        out_dir=out_dir,
        model_path=model_path,
        model_track=model_track,
        batch_size=batch_size,
        apply_negation=not no_negation,
        negation_window_tokens=negation_window,
        save_entity_spans=save_spans,
        debug=debug,
        debug_n_notes=debug_n_notes,
    )

    save_config_snapshot(
        cfg.__dict__ | {"pipeline_step": "extract"},
        run_dir=_current_run_dir(),
    )

    extractor = NERExtractor(cfg)
    extractor.load_model()
    extractor.run()


# ---------------------------------------------------------------------------
# 6. aggregate-features
# ---------------------------------------------------------------------------

@app.command(help="Aggregate per-note NER extractions to visit-level feature matrix.")
@log_timing
def aggregate_features(
    extractions_dir: Path = typer.Option(
        Path("data/interim/airms/ner_extractions"),
        help="Directory of NER extraction chunk Parquet files.",
    ),
    cohort_path: Path = typer.Option(
        Path("data/interim/airms/mrsa_cohort_person_list.parquet"),
        help="Cohort person list (PERSON_ID, MRN, LABEL).",
    ),
    level: str = typer.Option("visit", help="Aggregation level: 'visit' or 'person'."),
    include_negated: bool = typer.Option(
        True, "--include-negated/--no-include-negated",
        help="Include has_{entity}_negated features.",
    ),
    include_counts: bool = typer.Option(
        True, "--include-counts/--no-include-counts",
        help="Include count_{entity} features.",
    ),
    lookback: int = typer.Option(
        90,
        "--lookback",
        help="Pre-index lookback window in days for future leakage-gate filtering.",
    ),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
) -> None:
    """
    Pipeline Step 6 — NER feature engineering and aggregation.

    Aggregates per-note entity counts to visit level (MAX for binary features,
    SUM for count features), merges with cohort labels, and saves a
    training-ready CSV + Parquet.
    """
    _, run_dir = make_run_dir("ner_feature_aggregation")

    cfg = NERAggregatorConfig(
        extractions_dir=extractions_dir,
        cohort_person_list_path=cohort_path,
        out_dir=run_dir,
        aggregation_level=level,
        include_negated_features=include_negated,
        include_entity_counts=include_counts,
        lookback_days=lookback,
        debug=debug,
    )

    save_config_snapshot(cfg.__dict__ | {"pipeline_step": "aggregate_features"}, run_dir)

    agg = NERFeatureAggregator(cfg, run_dir)
    feature_df = agg.run()

    logger.info("NER feature matrix shape: %s", feature_df.shape if feature_df is not None else "None")


# ---------------------------------------------------------------------------
# 7. evaluate
# ---------------------------------------------------------------------------

@app.command(help="Evaluate NER extraction quality and generate visual reports.")
@log_timing
def evaluate(
    features_path: Path = typer.Argument(..., help="Path to the NER feature matrix CSV."),
    test_annotations: Optional[Path] = typer.Option(
        None, help="Held-out test annotation file (.spacy or CSV) for entity-level P/R/F1."
    ),
    rule_features_path: Optional[Path] = typer.Option(
        None, help="Rule-based feature matrix CSV for NER-vs-rules comparison (optional)."
    ),
    target_f1: float = typer.Option(0.70, help="Minimum per-entity-type F1 pass threshold."),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
) -> None:
    """
    Pipeline Step 7 — Evaluation and visualisation.

    Computes NER feature prevalence by LABEL.  If test annotations are
    provided, computes entity-level precision / recall / F1 using seqeval.
    Optionally compares NER features against rule-based features.
    Saves charts and a plain-text validation report.
    """
    _, run_dir = make_run_dir("ner_evaluation")

    cfg = NEREvaluatorConfig(
        features_path=features_path,
        test_annotations_path=test_annotations,
        rule_features_path=rule_features_path,
        out_dir=run_dir / "evaluation",
        target_f1=target_f1,
        debug=debug,
    )

    save_config_snapshot(cfg.__dict__ | {"pipeline_step": "evaluate"}, run_dir)

    evaluator = NEREvaluator(cfg, run_dir)
    evaluator.run()


# ---------------------------------------------------------------------------
# Full pipeline (run all steps in order)
# ---------------------------------------------------------------------------

@app.command(help="Run the complete NER pipeline end-to-end.")
@log_timing
def run_pipeline(
    schema: str = typer.Option("CDMPHI"),
    model_path: Path = typer.Option(Path("models/airms_ner_v1.0")),
    model_track: str = typer.Option("spacy"),
    skip_cohort: bool = typer.Option(False, "--skip-cohort", help="Skip cohort building (notes exist)."),
    skip_preprocess: bool = typer.Option(False, "--skip-preprocess"),
    skip_train: bool = typer.Option(False, "--skip-train", help="Skip training (use existing model)."),
    skip_extract: bool = typer.Option(False, "--skip-extract"),
    lookback: int = typer.Option(90, "--lookback", help="Pre-index lookback window in days."),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
) -> None:
    """
    Run all NER pipeline steps sequentially.

    Steps: build-cohort → preprocess → train → extract → aggregate-features
    Use --skip-* flags to resume from a specific step.
    """
    _, run_dir = make_run_dir("ner_full_pipeline")
    logger.info("Full NER pipeline run dir: %s", run_dir)

    if not skip_cohort:
        logger.info("=== Step 1/5: build-cohort ===")
        cfg_cohort = CohortConfig(schema=schema, debug=debug)
        conn = connect_hana()
        CohortBuilder(cfg_cohort, conn).run()

    if not skip_preprocess:
        logger.info("=== Step 2/5: preprocess ===")
        NERNotePreprocessor(NERPreprocessorConfig(debug=debug)).run()

    if not skip_train:
        logger.info("=== Step 3/5: train ===")
        schema_obj = AnnotationSchema(AnnotationSchemaConfig())
        NERModelTrainer(
            NERTrainerConfig(track=model_track, debug=debug),
            schema_obj,
            run_dir,
        ).run()

    if not skip_extract:
        logger.info("=== Step 4/5: extract ===")
        ext = NERExtractor(NERExtractorConfig(model_path=model_path, model_track=model_track, debug=debug))
        ext.load_model()
        ext.run()

    logger.info("=== Step 5/5: aggregate-features ===")
    NERFeatureAggregator(NERAggregatorConfig(debug=debug, lookback_days=lookback), run_dir).run()

    logger.info("NER pipeline complete.  Run dir: %s", run_dir)


# ---------------------------------------------------------------------------
# Mock Track A plumbing test
# ---------------------------------------------------------------------------

@app.command(help="Generate synthetic DocBins, train Track A, and evaluate on mock data.")
@log_timing
def mock_e2e(
    annotations_dir: Path = typer.Option(
        Path("annotations/mock"),
        help="Directory for synthetic train/val/test DocBins and sidecar CSVs.",
    ),
    model_out_dir: Path = typer.Option(
        Path("models/mock_airms_ner"),
        help="Directory for the mock-trained spaCy model.",
    ),
    base_model: str = typer.Option(
        "en_core_sci_sm",
        help="Preferred spaCy/scispaCy base model; falls back to spacy.blank('en') if unavailable.",
    ),
    n_epochs: int = typer.Option(
        5,
        help="Small epoch count for mock plumbing only; do not treat mock metrics as model quality.",
    ),
    batch_size: int = typer.Option(2, help="Mini-batch size for mock training."),
    dropout: float = typer.Option(0.2, help="Dropout for the mock training loop."),
    seed: int = typer.Option(GLOBAL_SEED, help=f"Random seed (default: {GLOBAL_SEED})."),
) -> None:
    """
    Run the local no-data Track A plumbing check.

    This command creates synthetic notes only. It does not connect to HANA,
    SSH, the cohort builder, Minerva, or any real patient data source.
    """
    run_dir = _current_run_dir()
    paths = generate_mock_ner_data(MockNERDataConfig(out_dir=annotations_dir, seed=seed))

    cfg = NERTrainerConfig(
        track="spacy",
        base_model=base_model,
        train_data_path=paths["train_docbin"],
        val_data_path=paths["val_docbin"],
        test_data_path=paths["test_docbin"],
        model_out_dir=model_out_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        dropout=dropout,
        eval_every_n_epochs=1,
        min_f1_to_save=0.0,
        early_stopping_patience=max(n_epochs + 1, 2),
        device="cpu",
        seed=seed,
        debug=False,
    )

    save_config_snapshot(
        {
            **cfg.__dict__,
            "pipeline_step": "mock_e2e",
            "mock_annotations_dir": annotations_dir,
            "mock_split_summary": paths["summary"],
        },
        run_dir,
    )

    schema = AnnotationSchema(AnnotationSchemaConfig())
    trainer = NERModelTrainer(cfg, schema, run_dir)
    trainer.run()
    logger.info("Mock Track A run complete. Run dir: %s", run_dir)


# ---------------------------------------------------------------------------
# Helper: retrieve current run dir set by configure_logging
# ---------------------------------------------------------------------------

def _current_run_dir() -> Path:
    """Return the run directory established by configure_logging."""
    from src.utils_logging import LOG_RUN_DIR
    return LOG_RUN_DIR or Path("outputs")


if __name__ == "__main__":
    if "extract-patient" not in sys.argv[1:]:
        configure_logging("INFO", run_name="cli")
    app()
