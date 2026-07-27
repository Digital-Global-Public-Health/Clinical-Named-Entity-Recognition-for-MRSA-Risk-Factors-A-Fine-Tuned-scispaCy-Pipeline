# MRSA NLP — NER-Based Pipeline

**Author:** Akhyar Ahmed

Fine-tuned neural named entity recognition (NER) for extracting MRSA clinical
risk signals from AIR.MS clinical notes.  Supports two training tracks:

- **Track A — scispaCy** (`en_core_sci_sm` fine-tuning, CPU-compatible)
- **Track B — BioClinicalBERT** (HuggingFace token classification, GPU recommended)

Produces the same visit-level binary feature matrix format as the rule-based
pipeline, enabling direct head-to-head comparison.

---
Training data format is defined in `../CONTRACT.md`. Treat that contract as the
source of truth for DocBin structure, entity labels, patient-level splits, and
sidecar attributes.

---

## Overview

The NER pipeline replaces hand-crafted regex patterns with a trained sequence
labelling model that learns contextual entity boundaries from annotated
clinical text.  It extracts three core entity types — `DISEASE`, `MEDICATION`,
`PROCEDURE` — applies a window-based negation detector, and aggregates
entity presence/count signals to visit level.

---

## Dataflow

```
mrsa_risk_predictions/
  data/interim/airms/
    mrsa_visit_cohort.parquet          ← shared cohort source (read-only)
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 · Cohort Builder  (src/cohort/cohort_builder.py)        │
│                                                                 │
│  · Load PERSON_ID + LABEL from mrsa_visit_cohort.parquet        │
│  · Query CDMPHI.PERSON → resolve MRNs                          │
│  · Save  data/interim/airms/mrsa_cohort_person_list.parquet     │
│  · Mine CDMPHI.NOTES in batches of 500 persons (resume-safe)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
          data/interim/airms/notes/
            chunk_0000.parquet  …
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 · NER Preprocessor  (src/preprocessing/note_preprocessor.py) │
│                                                                 │
│  · Normalise whitespace (lowercase=False for BERT)              │
│  · Expand clinical abbreviations                                │
│  · Filter by note length                                        │
│  · window_long_note(): split notes > max_tokens into            │
│    overlapping 512-token windows for BERT inference             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
          data/interim/airms/notes_preprocessed/
            chunk_0000.parquet  …
                         │
          ┌──────────────┤
          │              │
          ▼              ▼
   [annotated sample] [full notes]
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 · Annotation Preparation  (src/ner/annotation_schema.py) │
│                                                                 │
│  · Export Markdown annotation guidelines                        │
│  · Export schema.json for Label Studio / Prodigy               │
│  · Human annotators label 50–100 notes with:                    │
│      DISEASE / MEDICATION / PROCEDURE (/ SEVERITY optional)     │
│  → annotations/train.spacy  val.spacy  test.spacy               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4 · NER Model Training  (src/ner/model_trainer.py)        │
│                                                                 │
│  Track A — scispaCy fine-tuning                                 │
│    Base model : en_core_sci_sm  (scispaCy v0.5.4)               │
│    Framework  : spaCy 3.7                                       │
│    Training   : spaCy update loop + minibatch                   │
│    Evaluation : spaCy nlp.evaluate() → ents_f / ents_p / ents_r │
│    Device     : CPU                                             │
│    ~30 min on 100 annotated notes                               │
│                                                                 │
│  Track B — BioClinicalBERT fine-tuning                          │
│    Base model : emilyalsentzer/Bio_ClinicalBERT                 │
│    Framework  : HuggingFace Transformers 4.35+                  │
│    Task head  : AutoModelForTokenClassification (BIO tagging)   │
│    Evaluation : seqeval classification_report                   │
│    Device     : GPU (Minerva HPC recommended)                   │
│    ~2–4 h on 100 notes (A100 GPU)                               │
│                                                                 │
│  Both tracks:                                                   │
│    · Early stopping (patience=10)                               │
│    · Best checkpoint saved by validation F1                     │
│    · Training curves PNG + metrics.json                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
          models/airms_ner_v1.0/
            meta.json  (spaCy) or config.json (HF)
            model artifacts …
          outputs/train_spacy_YYYYMMDD-HHMMSS/
            training_curves.png
            metrics.json
            test_metrics.json
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5 · NER Extractor  (src/ner/ner_extractor.py)             │
│                                                                 │
│  · Load trained model from models/airms_ner_v1.0/               │
│  · For each note:                                               │
│      extract_entities_spacy()  or  extract_entities_hf()        │
│      detect_negation()  (5-token window NegEx)                  │
│      entities_to_features():                                    │
│        has_{DISEASE}  count_{DISEASE}  has_{DISEASE}_negated    │
│        has_{MEDICATION}  count_{MEDICATION}  …                  │
│        has_{PROCEDURE}  count_{PROCEDURE}  …                    │
│  · Resume-safe chunk processing                                 │
│  · Optional: save raw entity span text (--save-spans)           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
          data/interim/airms/ner_extractions/
            chunk_0000.parquet   (NOTE_ID | PERSON_ID | VISIT_OCCURRENCE_ID
            …                     | has_DISEASE | count_DISEASE |
                                   has_DISEASE_negated | …)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6 · NER Feature Aggregator  (src/features/feature_aggregator.py) │
│                                                                 │
│  · Aggregate per-note → visit level                             │
│      has_*          : MAX                                       │
│      count_*        : SUM                                       │
│      has_*_negated  : MAX                                       │
│  · Left-join with mrsa_cohort_person_list (adds LABEL + MRN)    │
│  · Log final case / control counts for verification             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
          outputs/ner_feature_aggregation_YYYYMMDD-HHMMSS/
            ner_features_<timestamp>.csv
            ner_features_<timestamp>.parquet
            ner_feature_summary_<timestamp>.json
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7 · Evaluator  (src/evaluation/evaluator.py)              │
│                                                                 │
│  · NER feature prevalence by LABEL (cases vs controls)          │
│  · If test annotations provided (test.spacy / test.csv):        │
│      seqeval span-level P / R / F1 per entity type              │
│      overall micro-averaged metrics                             │
│  · If rule features provided:                                   │
│      NER-vs-rules agreement analysis per feature                │
│      (both positive / NER-only / rules-only / neither)          │
│  · Validation report: pass/fail vs target F1 ≥ 0.70            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
          outputs/ner_evaluation_YYYYMMDD-HHMMSS/
            evaluation/
              ner_metrics_by_entity.png
              feature_prevalence.png
              ner_vs_rules_comparison.png  (if rule features given)
              label_distribution.png
              ner_metrics_by_entity.csv
              ner_vs_rules_comparison.csv
              validation_report.txt
```

---

## LLM Pre-Annotation Drafts

The pre-annotation stage proposes spans with a local Ollama model, verifies
every proposed string against the source note text, and writes real character
offsets. These are first-draft annotations for human review in INCEpTION, not
gold labels.

Offline WSL smoke test with synthetic notes and canned responses:

```bash
python -m src.cli preannotate \
  --synthetic-fixtures \
  --out-dir annotations/preannotations/synthetic \
  --overwrite
```

Real Minerva run, after `.env` contains `OLLAMA_HOST`,
`OLLAMA_AUTH_USER`, `OLLAMA_AUTH_TOKEN`, and either `OLLAMA_MODEL` or
`--model` is provided:

```bash
python -m src.cli preannotate \
  --input-path data/interim/airms/notes_preprocessed \
  --out-dir annotations/preannotations/minerva \
  --model "$OLLAMA_MODEL"
```

Primary artifacts are written under `verified/` as one JSON file per note.
Derived review/training exports are written under `inception_webanno_tsv3/`
and `contract_docbin/`.

---

## Project Structure

```
ner_based/
├── env/
│   └── environment.yml          # conda env: mrsa-nlp-ner (Python 3.10)
├── annotations/                 # training data (human-annotated)
│   ├── train.spacy              # spaCy DocBin format (or train.csv)
│   ├── val.spacy
│   ├── test.spacy
│   └── annotation_guidelines.md # generated by prepare-annotations
├── models/
│   └── airms_ner_v1.0/          # saved model (Track A: spaCy, Track B: HF)
├── scripts/
│   ├── start_airms_tunnel.sh
│   ├── run_cohort_builder.sh
│   ├── run_preprocessing.sh
│   ├── run_training.sh          # supports --track spacy|hf
│   ├── run_feature_extraction.sh
│   └── run_evaluation.sh
├── src/
│   ├── cli.py                   # Typer CLI — 8 commands
│   ├── utils_logging.py
│   ├── utils_db.py
│   ├── utils_io.py
│   ├── cohort/
│   │   └── cohort_builder.py
│   ├── preprocessing/
│   │   └── note_preprocessor.py # NERPreprocessorConfig + NERNotePreprocessor
│   ├── ner/
│   │   ├── annotation_schema.py # entity type definitions + guideline export
│   │   ├── model_trainer.py     # NERTrainerConfig + NERModelTrainer
│   │   └── ner_extractor.py     # NERExtractorConfig + EntitySpan + NERExtractor
│   ├── features/
│   │   └── feature_aggregator.py
│   └── evaluation/
│       └── evaluator.py
├── data/
│   └── interim/airms/
│       ├── notes/
│       ├── notes_preprocessed/
│       └── ner_extractions/
├── outputs/
├── .env.example
└── .gitignore
```

---

## Models

### Track A — scispaCy (`en_core_sci_sm`)

| Property | Value |
|---|---|
| Model | `en_core_sci_sm` v0.5.4 |
| Publisher | Allen Institute for AI (AllenAI) |
| Base | spaCy `en_core_web_sm` + biomedical NER pre-training |
| Vocabulary | ~360k biomedical tokens |
| Training data | MedMentions + BC5CDR |
| Fine-tuned on | 50–100 annotated AIR.MS notes |
| Framework | spaCy 3.7 |
| Device | CPU |
| Install | `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz` |

**When to use:** fast iteration, no GPU, smaller annotated sets.

### Track B — BioClinicalBERT

| Property | Value |
|---|---|
| Model | `emilyalsentzer/Bio_ClinicalBERT` |
| Publisher | Emily Alsentzer, Harvard / MIT |
| Base | `bert-base-uncased` → BioBERT → MIMIC-III notes fine-tuning |
| Training data | All MIMIC-III clinical notes |
| Fine-tuned on | 50–100 annotated AIR.MS notes (BIO token classification) |
| BIO scheme | `B-DISEASE`, `I-DISEASE`, `B-MEDICATION`, `I-MEDICATION`, `B-PROCEDURE`, `I-PROCEDURE`, `O` |
| Evaluation | seqeval (span-level, strict matching) |
| Framework | HuggingFace Transformers 4.35+ |
| Device | GPU (NVIDIA A100 on Minerva recommended) |
| HF Hub | `emilyalsentzer/Bio_ClinicalBERT` |

**When to use:** maximum accuracy, GPU available, final production model.

### Alternative HF models (swap via `--base-model`)

| Model | Notes |
|---|---|
| `allenai/biomed_roberta_base` | RoBERTa pre-trained on PubMed + PMC |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | PubMed abstract pre-training |
| `d4data/biomedical-ner-all` | NER fine-tuned on multiple biomedical corpora |
| `en_core_sci_lg` | scispaCy large model (better accuracy, more RAM) |

---

## Setup

### 1 — Create the conda environment

```bash
cd mrsa_nlp/ner_based
conda env create -f env/environment.yml
conda activate mrsa-nlp-ner
```

> **GPU note (Track B):** if running on Minerva, load the CUDA module first:
> ```bash
> module load cuda/11.8
> conda env create -f env/environment.yml
> ```

### 2 — Configure credentials

```bash
cp .env.example .env
# Edit .env — fill AIRMS_USER, AIRMS_PASSWORD, AIRMS_PORT
```

### 3 — Database connection (automatic via HPC login node)

The SSH tunnel to AIR.MS is established **automatically** through the HPC login node.

No separate setup needed — just run:

```bash
bash scripts/run_cohort_builder.sh
```

The script will:
1. Prompt for your HANA password
2. Find a free local port (50000–51000 range)
3. Establish SSH tunnel: `localhost:PORT → li04e02 → db.airms.mssm.edu:30041`
4. Set all connection environment variables
5. Run cohort builder
6. Clean up tunnel automatically on exit

**Why this approach:**
- ✓ Works from local machines (via HPC login node)
- ✓ No need for separate terminal running tunnel
- ✓ Automatic port discovery avoids conflicts
- ✓ SSH stability flags (`ServerAliveInterval=60`, `ExitOnForwardFailure`)

---

## Running the Pipeline

### Complete walkthrough (Track A — scispaCy)

```bash
conda activate mrsa-nlp-ner
cd mrsa_nlp/ner_based

# 1. Build cohort + mine notes (requires tunnel)
bash scripts/run_cohort_builder.sh

# 2. Preprocess
bash scripts/run_preprocessing.sh

# 3. Export annotation guidelines, then annotate notes externally
python -m src.cli prepare-annotations \
    --entity-types "DISEASE,MEDICATION,PROCEDURE" \
    --guidelines-out annotations/annotation_guidelines.md \
    --schema-json-out annotations/schema.json
#  → open annotations/ in Label Studio or Prodigy
#  → export annotated data as train.spacy / val.spacy / test.spacy

# 4. Train (Track A, CPU)
bash scripts/run_training.sh --track spacy

# 5. Run NER extraction + aggregate features
bash scripts/run_feature_extraction.sh --track spacy

# 6. Evaluate
bash scripts/run_evaluation.sh \
    outputs/ner_feature_aggregation_<timestamp>/ner_features_<timestamp>.csv \
    --test-annotations annotations/test.spacy
```

### Track B — BioClinicalBERT (GPU)

```bash
# Steps 1–3 same as above, then:

# 4. Train on GPU
bash scripts/run_training.sh --track hf

# 5. Extract + aggregate
bash scripts/run_feature_extraction.sh --track hf

# 6. Evaluate + compare with rule-based features
bash scripts/run_evaluation.sh \
    outputs/ner_feature_aggregation_<timestamp>/ner_features_<timestamp>.csv \
    --test-annotations annotations/test.spacy \
    --rule-features    ../../rule_based/outputs/feature_aggregation_<timestamp>/rule_features_<timestamp>.csv
```

### Debug mode (no GPU, 20 persons)

```bash
bash scripts/run_cohort_builder.sh --debug
bash scripts/run_preprocessing.sh --debug
bash scripts/run_training.sh --track spacy --debug
bash scripts/run_feature_extraction.sh --debug
```

---

## CLI Reference

```
python -m src.cli --help
```

```
 Usage: python -m src.cli [OPTIONS] COMMAND [ARGS]...

 MRSA NLP — NER-based clinical note extraction and model training pipeline.

Options:
  --log-level TEXT  Logging level: DEBUG | INFO | WARNING | ERROR  [default: INFO]
  --help            Show this message and exit.

Commands:
  build-cohort         Load MRSA cohort and mine notes from CDMPHI.NOTES
  preprocess           Clean and normalise raw note chunks for NER
  prepare-annotations  Export annotation schema and guidelines
  train                Train or fine-tune a NER model
  extract              Run NER inference on preprocessed note chunks
  aggregate-features   Aggregate NER extractions to visit-level matrix
  evaluate             Evaluate NER quality and generate reports
  run-pipeline         Run the complete pipeline end-to-end
```

### `build-cohort`

```bash
python -m src.cli build-cohort \
    --schema CDMPHI \
    --chunk-size 500 \
    --min-note-date 2014-07-14 \
    --no-debug
```

### `preprocess`

```bash
# Track A (spaCy) — no windowing needed
python -m src.cli preprocess \
    --raw-notes-dir data/interim/airms/notes \
    --out-dir       data/interim/airms/notes_preprocessed \
    --max-tokens 0 \
    --expand-abbrev \
    --no-debug

# Track B (BERT) — window notes to 512 tokens
python -m src.cli preprocess \
    --raw-notes-dir data/interim/airms/notes \
    --out-dir       data/interim/airms/notes_preprocessed \
    --max-tokens 512 \
    --expand-abbrev
```

### `prepare-annotations`

```bash
# Core 3 entity types
python -m src.cli prepare-annotations \
    --entity-types "DISEASE,MEDICATION,PROCEDURE" \
    --guidelines-out annotations/annotation_guidelines.md

# Include optional SEVERITY type
python -m src.cli prepare-annotations \
    --entity-types "DISEASE,MEDICATION,PROCEDURE" \
    --include-severity \
    --guidelines-out  annotations/annotation_guidelines.md \
    --schema-json-out annotations/schema.json
```

### `train`

```bash
# Track A — scispaCy fine-tuning (CPU)
python -m src.cli train \
    --track       spacy \
    --base-model  en_core_sci_sm \
    --train-data  annotations/train.spacy \
    --val-data    annotations/val.spacy \
    --test-data   annotations/test.spacy \
    --model-out-dir models/airms_ner_v1.0 \
    --n-epochs    30 \
    --batch-size  16 \
    --dropout     0.3 \
    --device      cpu \
    --seed        42

# Track B — BioClinicalBERT (GPU)
python -m src.cli train \
    --track       hf \
    --base-model  emilyalsentzer/Bio_ClinicalBERT \
    --train-data  annotations/train.spacy \
    --val-data    annotations/val.spacy \
    --test-data   annotations/test.spacy \
    --model-out-dir models/airms_ner_v1.0 \
    --n-epochs    10 \
    --batch-size  8 \
    --learning-rate 5e-5 \
    --device      cuda:0
```

| Option | Track A default | Track B default |
|---|---|---|
| `--n-epochs` | 30 | 10 |
| `--batch-size` | 16 | 8 |
| `--learning-rate` | `1e-3` | `5e-5` |
| `--device` | `cpu` | `cuda:0` |

### `extract`

```bash
python -m src.cli extract \
    --preprocessed-dir data/interim/airms/notes_preprocessed \
    --out-dir          data/interim/airms/ner_extractions \
    --model-path       models/airms_ner_v1.0 \
    --model-track      spacy \
    --batch-size       32 \
    --negation-window  5 \
    --no-debug
```

### `aggregate-features`

```bash
python -m src.cli aggregate-features \
    --extractions-dir data/interim/airms/ner_extractions \
    --cohort-path     data/interim/airms/mrsa_cohort_person_list.parquet \
    --level           visit \
    --include-negated \
    --include-counts
```

### `evaluate`

```bash
# Prevalence + entity metrics + NER-vs-rules comparison
python -m src.cli evaluate \
    outputs/ner_feature_aggregation_20250401-150000/ner_features_20250401-150000.csv \
    --test-annotations annotations/test.spacy \
    --rule-features-path ../../rule_based/outputs/feature_aggregation_<ts>/rule_features_<ts>.csv \
    --target-f1 0.70
```

---

## Entity Types

| Label | Examples | MRSA relevance |
|---|---|---|
| `DISEASE` | pneumonia, MRSA, UTI, sepsis, diabetes, CKD, HIV, lymphoma, cellulitis | Comorbidities + prior infections are key risk factors |
| `MEDICATION` | prednisone, vancomycin, methotrexate, tacrolimus, ciprofloxacin, corticosteroids | Immunosuppressants + prior antibiotics |
| `PROCEDURE` | central line, PICC, hemodialysis, intubation, surgery, bone marrow transplant | Invasive procedures = primary acquisition routes |
| `SEVERITY` *(optional)* | immunocompromised, neutropenic, critically ill, debilitated | Cross-cutting immune status modifier |

**BIO tagging scheme (Track B):**

```
Text:    The patient has  a  central    line   and  is  on  prednisone  .
BIO:     O   O       O    O  B-PROCEDURE I-PROCEDURE O O  O  B-MEDICATION O
```

---

## Outputs

```
models/
  airms_ner_v1.0/               ← final trained model

outputs/
  train_spacy_20250401-100000/
    training_curves.png         ← loss + val F1 twin-axis plot
    metrics.json                ← per-epoch training metrics
    test_metrics.json           ← final held-out test scores
    checkpoints/
      epoch_05/  epoch_10/ …    ← spaCy nlp.to_disk() snapshots
    run.log
    config.yaml

  ner_feature_aggregation_20250401-150000/
    ner_features_<timestamp>.csv
    ner_features_<timestamp>.parquet
    ner_feature_summary_<timestamp>.json

  ner_evaluation_20250401-160000/
    evaluation/
      ner_metrics_by_entity.png   ← P/R/F1 grouped bar chart by entity type
      feature_prevalence.png      ← prevalence by LABEL (case vs control)
      ner_vs_rules_comparison.png ← stacked agreement bars (if rule features given)
      label_distribution.png
      ner_metrics_by_entity.csv
      ner_vs_rules_comparison.csv
      validation_report.txt       ← pass/fail vs target F1
    run.log
    config.yaml
```

---

## Negation Logic

Shared with the rule-based pipeline; applied post-entity-extraction:

```
For each extracted EntitySpan [start, end]:
  1. Tokenise the window [start - 5 tokens, start]
  2. Check for negation cue:
       no · not · without · denies · denied · negative for ·
       no evidence of · no sign of · ruled out · absent · never …
  3. If sentence-boundary mode (default True):
       truncate window at nearest preceding sentence boundary
  4. Mark EntitySpan.is_negated = True if cue found

Downstream:
  is_negated=False  → contributes to has_{entity} and count_{entity}
  is_negated=True   → contributes to has_{entity}_negated only
```

---

## Key Design Decisions

- **`lowercase=False` in preprocessing** — BERT models are case-sensitive; preserving capitalisation improves entity recognition (e.g. "MRSA" vs "mrsa").
- **BERT 512-token windowing** — long clinical notes are split into overlapping windows with a configurable stride; entities near window edges are deduplicated.
- **BIO tagging, not IO** — `B-` prefix distinguishes adjacent same-type entities (e.g. two consecutive disease mentions).
- **seqeval for evaluation** — span-level strict matching (both boundaries + label must match); reports per-class and micro-averaged P/R/F1.
- **Same cohort as `mrsa_risk_predictions`** — reads `mrsa_visit_cohort.parquet` directly; same persons, same labels.
- **Track A first** — recommended starting point: faster iteration, no GPU needed, interpretable failure modes.
- **Track B for production** — BioClinicalBERT is pre-trained on MIMIC-III, making it domain-adapted to clinical notes similar to AIR.MS.

---

## Annotation Workflow

```
1. python -m src.cli prepare-annotations
   → annotations/annotation_guidelines.md   (human-readable)
   → annotations/schema.json                (for Label Studio / Prodigy)

2. Sample 50–100 notes from data/interim/airms/notes_preprocessed/

3. Annotate in Label Studio (recommended):
     label_studio start
     # Import schema.json as label config
     # Import note texts as tasks
     # Export as spaCy format → train.spacy / val.spacy / test.spacy

   Or use Prodigy:
     prodigy ner.manual airms_ner_train \
         en_core_sci_sm \
         path/to/notes.jsonl \
         --label DISEASE,MEDICATION,PROCEDURE

4. Recommended split: 70% train / 15% val / 15% test
   Aim for ≥ 200 annotated entity spans per type.

5. Run: python -m src.cli train --track spacy
```

---

## References

### Models

- Neumann M et al. (2019). *ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing.* BioNLP workshop at ACL. — scispaCy base model.
- Alsentzer E et al. (2019). *Publicly Available Clinical BERT Embeddings.* Clinical NLP workshop at NAACL. — BioClinicalBERT model trained on MIMIC-III.
- Lee J et al. (2020). *BioBERT: a pre-trained biomedical language representation model.* Bioinformatics. — Biomedical BERT pre-training methodology.
- Devlin J et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL. — Original BERT architecture.

### Evaluation

- Ramshaw LA & Marcus MP (1995). *Text chunking using transformation-based learning.* — BIO tagging scheme.
- Nakayama H (2018). *seqeval: A Python framework for sequence labeling evaluation.* — span-level NER metrics.

### Negation

- Chapman WW et al. (2001). *A simple algorithm for identifying negated findings and diseases in discharge summaries.* Journal of Biomedical Informatics.

### Clinical NLP

- Peng Y et al. (2019). *Transfer learning in biomedical NLP: an evaluation of BERT and ELMo on ten benchmarking datasets.* BioNLP workshop. — Transfer learning evaluation.
- Soysal E et al. (2018). *CLAMP — a toolkit for efficiently building customized clinical NLP pipelines.* JAMIA.
