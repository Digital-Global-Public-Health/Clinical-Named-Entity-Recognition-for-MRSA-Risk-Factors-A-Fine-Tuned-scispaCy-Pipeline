# Repository inventory

What each file in `ner_based/` is, and whether it is part of the pipeline, a
diagnostic, an applied one-off, or inherited scaffold.

This repository contains two codebases that barely touch:

- **The thesis pipeline.** LLM teacher pre-annotation -> INCEpTION -> scispaCy
  student model -> medspaCy assertion layer -> lexicon feature matrix. Mostly
  root-level scripts plus `src/ner/`, `configs/`, and `scripts/train_ner.sh`.
- **An inherited scaffold.** A Track A / Track B design (`src/cli.py`,
  `src/cohort/`, `src/preprocessing/`, `src/features/`, `src/evaluation/`, and
  the `scripts/run_*.sh` wrappers) that predates the thesis work. Four of its
  modules are specification-only: every method body is `pass`. It cannot be
  removed, because `src/cli.py` imports all of it and `src/cli.py` carries two
  live pipeline stages. See section D.

---

## A. Pipeline components, in run order

| # | Stage | Script / module | Reads | Writes |
|---|---|---|---|---|
| 0 | Corpus extraction | `extract_cohort_all.py` -> `src/ner/extract_patient.py` | rule-based `cohort_notes.parquet`, `cohort_subset.csv` | `cohort_all_notes.parquet` |
| 1 | LLM pre-annotation (teacher) | `python -m src.cli preannotate` -> `src/ner/preannotate.py` | note parquet/JSON/JSONL/TXT, Ollama | `annotations/<batch>/verified/*.json`, `run_summary.json` |
| 1b | Review / training exports | `src/ner/preannotation_serializers.py` (same command) | `verified/*.json` | `inception_webanno_tsv3/*.tsv`, `contract_docbin/preannotated.spacy` + attribute sidecar + `alignment_failures.jsonl` |
| 2 | Load into INCEpTION | `import_preannotations.py` | WebAnno TSV3 directory | documents POSTed to INCEpTION via the AERO API |
| 2b | Assertion annotation layer | `rewrite_tsv_layer.py` | project-4 TSV exports | TSVs re-typed onto `webanno.custom.Assertion` with five features |
| 3 | Gold set -> DocBin | `inception_to_docbin.py` | INCEpTION project export `.zip` | `gold.spacy`, optional `gold_spans.csv` |
| 3b | Gold assertion attributes | `project6_to_spans.py` | `goldassert_*.zip` | `/tmp/gold6_spans.csv` |
| 4 | Silver corpus splits | `build_splits.py` | `verified/*.json` + notes parquet | `splits/{train,dev,test}.spacy`, `train_{2000,5000,10000}.spacy`, `manifest.csv`, `gold_candidates.csv`, `split_report.md` |
| 5 | Student training | `scripts/train_ner.sh {2000\|5000\|10000\|full}` -> `configs/ner_sci.cfg`, `configs/custom_code.py` | `splits/train_*.spacy`, `splits/dev_500.spacy` | `models/ner_<size>/model-best` |
| 6 | NER evaluation | `eval_gold.py` (models vs gold), `eval_teacher.py` (teacher ceiling) | `gold.spacy`, models or `verified/*.json` | stdout; captured in `docs/eval_2026-08-30_*.txt` |
| 7 | Assertion layer | `src/ner/assertion.py` + `src/ner/allergy.py` | a trained model directory | an in-process spaCy pipeline; no artefact |
| 7b | Assertion evaluation | `score_assertions.py` | project-6 export dir, model, gold note text | stdout, optional `--errors-csv` |
| 8 | Feature matrix | `build_feature_matrix.py` + `lexicon.yaml` | notes parquet, model | note-level counts CSV; `--from-counts --level visit\|patient` derives the binary view |

Supporting, outside the sequence:

- `review_lexicon.py` — the curation loop that produced `lexicon.yaml`.
- `measure_duplicates.py` — its `--out-csv` optionally feeds `build_splits.py --dup-csv`.

## B. One-off scripts, already applied

These are history. Their effects are already in `lexicon.yaml`, in
`src/ner/assertion.py`, or in decisions the pipeline now encodes. They are kept
for provenance and are not build steps. They live in `archive/`.

| Script | What it did | Where the result lives now |
|---|---|---|
| `patch_lexicon_01.py` | Five fixes from the first corpus review: drop CAD patterns from `myocardial_infarction`, tighten `surgery`, add `dlbcl` and `gvhd`, drop bare `ALL` | `lexicon.yaml` |
| `patch_lexicon_02.py` | Seven fixes from the second review: exclude `hd-mtx` and midline catheters, drop CPAP/BiPAP from `supplemental_oxygen`, add `pna`, `vanc`, `olt`, `ddkt`, `cabg`, `pci` | `lexicon.yaml` |
| `patch_lexicon_02b.py` | Allow PROCEDURE spans to match `immunosuppressed_state` | `lexicon.yaml` |
| `patch_lexicon_03.py` | Added feature 48 `other_indwelling_device`; moved `oht` to `immunosuppressed_state`; acronym sweep | `lexicon.yaml` |
| `patch_assertion_packaged_rules.py` | Constrained two mis-firing packaged medspaCy rules (`prophylaxis`, `: no`) | `PACKAGED_RULE_FIXES` in `src/ner/assertion.py` |
| `reverify.py` | Replayed stored proposals through the fixed word-boundary matcher, holding model output constant | `annotations/batch01_v2/` |
| `compare_runs.py` | Span-overlap comparison of `llama3.1:70b` against `gemma3:27b` pre-annotations | the choice of the llama family as teacher; the production run used `llama3.3:70b` |
| `procedure_gap.py` | The same comparison restricted to PROCEDURE spans | the same choice |
| `check_reuse.py` | Which already-annotated pilot notes could be reused when extending the gold set | the 16-note gold set composition |
| `draw_new_gold.py` | Drew the ten extension notes (seed 7, four patients x four note types) | `splits/new_gold_10.csv`, the 16-note gold set |
| `check_recommender_overlap.py` | Measured how many gold spans came from accepted INCEpTION recommender suggestions | the pilot/extension split reported by `eval_gold.py --split` |
| `test_prompt_variants.py` | Tested prompt reframing vs multi-pass union against the 6-note gold set | the production prompt in `src/ner/preannotate.py` |

## C. Diagnostics

Re-runnable analysis tools. They read pipeline artefacts and print or sample;
none produces an input to a later stage.

| Script | Question it answers |
|---|---|
| `profile_cohort.py` | Is a patient-level 70/15/15 split with all note types in every split feasible at all? |
| `measure_duplicates.py` | Exact and near-duplicate rate in the corpus; do near-duplicate clusters cross patients? |
| `measure_allergy_sections.py` | How often does a MEDICATION span sit under an allergy header? (The 6.3% / 1,330-note figure quoted in `src/ner/allergy.py`.) |
| `sample_allergy.py` | Blind adjudication samples for allergy precision and recall |
| `review_lexicon.py` | Per-feature coverage, matched terms, and the unmapped-term gap list |
| `teacher_vs_gold.py` | Teacher vs gold with a disagreement CSV (`--errors-csv`) |
| `analyse_teacher_misses.py` | Are the teacher's misses occurrence-density artefacts or systematic string failures? |
| `analyse_assert_errors.py` | Assertion disagreements broken down by note type, label, and cue |
| `which_cue.py` | Which ConText cue produced each hypothetical false positive |
| `which_cue_exp.py` | The same, for experiencer false positives |
| `scripts/assertion_report.py` | Per-entity assertion flags with context, as a review CSV |
| `scripts/assertion_sample.py` | Note-stratified manual-review sample drawn from that CSV |
| `scripts/ablate_context_rules.py` | Flag counts with and without the AIR.MS cue list |
| `scripts/generate_mock_ner_data.py` | Synthetic DocBins for the offline plumbing check |

## D. Inherited scaffold

### Specification-only modules

Every method body is `pass` beneath a docstring describing intended behaviour.
Nothing here has been executed against real data.

- `src/cohort/cohort_builder.py`
- `src/preprocessing/note_preprocessor.py`
- `src/features/feature_aggregator.py`
- `src/evaluation/evaluator.py`

The wrappers that drive them run to completion and do nothing:
`scripts/run_cohort_builder.sh`, `scripts/run_preprocessing.sh`,
`scripts/run_feature_extraction.sh`, `scripts/run_evaluation.sh`, and
`scripts/start_airms_tunnel.sh`. `src/cli.py build-cohort` tolerates the `None`
return and logs `Cohort built: 0 persons`, exiting 0.

### Implemented but superseded

- `src/ner/model_trainer.py` — Track A (spaCy) is implemented and backs the
  `mock-e2e` plumbing command. Track B raises
  `NotImplementedError("Track B / HuggingFace training is out of thesis scope")`.
  The thesis models came from `spacy train`, not from this module.
- `src/ner/ner_extractor.py` — the spaCy path is implemented; the HF path and
  `detect_negation` raise `NotImplementedError`. Superseded by
  `build_feature_matrix.py`, which runs the assertion pipeline directly.
- `src/ner/annotation_schema.py` — implemented. Its `export_guidelines()` writes
  a short generated document to `annotations/annotation_guidelines.md`. This is
  **not** `docs/annotation_guidelines.md`, which is the hand-written thesis
  document.
- `src/ner/mock_data.py` — synthetic notes for the offline plumbing check.

### Why none of it can be deleted

`src/cli.py` imports all fourteen `src/` modules at module load, and two of its
commands are live pipeline stages:

- `preannotate` — stage 1, the teacher run.
- `extract-patient` — the single-patient path of stage 0.

Removing any scaffold module breaks both.

### Live infrastructure

`src/utils_logging.py` (run directories, config snapshots, `@log_timing`),
`src/utils_seed.py` (`GLOBAL_SEED = 7`), `src/utils_io.py`,
`src/utils_db.py` (HANA connection from environment variables).

## E. Tests

`tests/test_extract_patient.py`, `tests/test_preannotate.py`,
`tests/test_preannotation_serializers.py`. They cover the two live `src/ner/`
stages and the span verifier. Nothing covers the assertion layer or the feature
matrix.
