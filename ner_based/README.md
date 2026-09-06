# MRSA NER — clinical named entity recognition for MRSA risk factors

**Author:** Tobias Rademacher

A distillation pipeline for extracting MRSA risk-factor signals from AIR·MS
clinical notes. A local LLM ("teacher") pre-annotates a large silver corpus, a
fine-tuned scispaCy model ("student") is trained on it, a medspaCy ConText layer
adds assertion attributes at inference time, and a curated lexicon maps the
resulting entities onto a binary risk-factor feature matrix.

A 16-note hand-annotated gold set measures both the teacher and the student.

> **Everything runs inside the hospital HPC enclave.** All note text is PHI.
> Data, models and outputs are gitignored and never leave the enclave. See
> [Reproducibility](#what-can-and-cannot-be-reproduced) for what an outside
> reader can and cannot rebuild.

Companion documents:

- [`docs/inventory.md`](docs/inventory.md) — what every file in this directory is.
- [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md) — the annotation standard.
- [`docs/assertion.md`](docs/assertion.md) — the assertion layer's review procedure and limitations.
- [`../CONTRACT.md`](../CONTRACT.md) — the training-data format contract.
- [`archive/README.md`](archive/README.md) — one-off scripts already applied.

---

## What the pipeline does

```
cohort notes (parquet, enclave)
        │
        ▼  src/cli.py preannotate
llama teacher proposes entity strings; every string is verified against the
source text before offsets are written                    → verified/*.json
        │
        ├──▶ WebAnno TSV3 ──▶ INCEpTION ──▶ human gold set (16 notes)
        │                                          │
        ▼                                          ▼  inception_to_docbin.py
   build_splits.py                              gold.spacy
   patient-level 70/15/15                          │
        │                                          │
        ▼  scripts/train_ner.sh                    │
   scispaCy student models                         │
   models/ner_{2000,5000,10000,full}               │
        │                                          │
        ├──────────────────────────────────────────┤
        │                                          ▼
        │                          eval_gold.py / eval_teacher.py
        ▼  src/ner/assertion.py + allergy.py
   medspaCy ConText: negated / historical / hypothetical / uncertain /
   family / allergy                              → score_assertions.py
        │
        ▼  build_feature_matrix.py + lexicon.yaml
   note-level counts → visit- or patient-level binary feature matrix
```

Three entity labels are in scope: `DISEASE`, `MEDICATION`, `PROCEDURE`. Gold
annotation is spans and labels only; assertion attributes are applied after NER
at inference time, never baked into `doc.ents`. `../CONTRACT.md` is the
authority on this.

---

## Stages

Run everything from `ner_based/`.

### 0 — Corpus extraction

```bash
python extract_cohort_all.py
```

Reads the shared rule-based cohort (`cohort_notes.parquet`, `cohort_subset.csv`)
and writes one parquet of every note for every cohort patient. Paths are
hardcoded to the enclave; the script prints the note total and a breakdown by
`NOTE_TITLE`. The 50-patient cohort yields ~20,951 notes.

Single-patient extraction, used for pre-annotation smoke tests, goes through the
CLI instead:

```bash
python -m src.cli extract-patient \
    --notes-parquet <cohort_notes.parquet> \
    --cohort-csv    <cohort_subset.csv> \
    --person-id     "$PERSON_ID" \
    --output        "$EXTRACTED_NOTES"
```

Both call `src/ner/extract_patient.py`, which filters at read time and refuses
person IDs absent from the cohort CSV.

### 1 — LLM pre-annotation (the teacher)

```bash
python -m src.cli preannotate \
    --input-path <notes parquet or dir> \
    --out-dir    annotations/<batch>/ \
    --model      "$OLLAMA_MODEL"
```

`src/ner/preannotate.py` asks a local Ollama model for candidate entity strings
and treats the response strictly as proposals. **The source text is
authoritative:** every accepted span is located in the note by exact match, or
by whitespace-normalised match, with word-boundary guards so `MI` does not match
inside `MIRALAX`. Anything that cannot be located is rejected and recorded with
a reason (`not_in_text` vs `no_boundary_match`, which distinguishes a
hallucination from a substring collision).

Requires `OLLAMA_HOST`, `OLLAMA_AUTH_USER`, `OLLAMA_AUTH_TOKEN` and either
`OLLAMA_MODEL` or `--model` in the environment or a `.env` file.
`PREANNOTATE_WORKERS` sets thread count (default 1). The run is resume-safe: a
note with an existing artifact is skipped unless `--overwrite` is given, and a
failed request writes no artifact at all, so the next run retries it.

Writes, per note, to `<out-dir>/verified/<note_id>.json`: the note text, the
accepted spans with character offsets, the rejected proposals with reasons, and
per-note verification counters. Plus `run_summary.json` for the batch.

Two derived exports are written by `src/ner/preannotation_serializers.py` in the
same command:

- `inception_webanno_tsv3/` — one WebAnno TSV 3.3 file per note, on the built-in
  DKPro `NamedEntity` layer, for import into INCEpTION.
- `contract_docbin/` — a `CONTRACT.md`-shaped DocBin plus an empty reserved
  attribute sidecar, and `alignment_failures.jsonl` for spans spaCy could not
  align.

Offline check, no Ollama server and no PHI:

```bash
python -m src.cli preannotate --synthetic-fixtures \
    --out-dir annotations/preannotations/synthetic --overwrite
```

### 2 — Human annotation in INCEpTION

```bash
python import_preannotations.py \
    --base-url <local INCEpTION> --user <u> --password <p> \
    --project-id <n> --tsv-dir <batch>/inception_webanno_tsv3
```

Pushes the TSVs through INCEpTION's AERO remote API. **Run on a compute node
against the local server** so no note text leaves the enclave.

For the assertion gold pass, the spans must first be moved off the built-in
`NamedEntity` layer, which INCEpTION does not allow features to be added to:

```bash
python rewrite_tsv_layer.py --in-dir gold_tsv --out-dir gold_tsv_assert
```

`rewrite_tsv_layer.py` re-types them onto `webanno.custom.Assertion` with five
features (`polarity`, `certainty`, `temporality`, `experiencer`, `allergy`) and
bakes in the default values, so the annotator only touches exceptions.

Annotation follows `docs/annotation_guidelines.md`.

### 3 — Gold set to DocBin

```bash
python inception_to_docbin.py \
    --zip      <export>.zip \
    --out      annotations/gold_export/gold.spacy \
    --spans-csv annotations/gold_export/gold_spans.csv
```

Reads the INCEpTION export zip directly, so nothing is unpacked into the repo.
Tokenises with `en_core_sci_sm` so the gold DocBin aligns with the trained
models. Documents with zero annotations are skipped by default — in a partly
annotated project those are notes that were opened but never worked on, and
including them would destroy the recall figures.

The assertion-attribute export is parsed separately by `project6_to_spans.py`,
which writes a span CSV carrying all five attribute columns.

### 4 — Silver corpus splits

```bash
python build_splits.py \
    --ann-dirs annotations/batch01_v2 annotations/batch02 annotations/batch03 \
    --parquet  <cohort_all_notes.parquet> \
    --out-dir  splits
```

Rebuilds DocBins from the verified JSONs rather than merging per-batch DocBins,
so filtering decisions are explicit and identical across splits. Requirements,
in the order the script enforces them:

1. **Hard** — patient integrity: no patient's notes straddle a split.
2. **Optimised** — 70/15/15 by note count, greedy longest-patient-first.
3. **Tie-break** — case/control balance.
4. **Asserted** — all four note types present in every split; the script exits
   non-zero if not.
5. **Fixed** — seed 7, deterministic; no searching for a lucky partition.

Writes `train.spacy`, `dev.spacy`, `test.spacy`, nested learning-curve subsets
`train_{2000,5000,10000}.spacy`, `manifest.csv` (the join key for everything),
`gold_candidates.csv`, and `split_report.md`.

The committed [`splits/split_report.md`](splits/split_report.md) records the
actual run: 18,669 notes ≥ 200 chars across 50 patients, 571,455 silver
entities, 70.0 / 15.0 / 15.0.

`manifest.csv` and `gold_candidates.csv` carry `PERSON_ID` and `NOTE_ID` and
stay on the enclave. `split_report.md` is aggregate counts only and is committed.

### 5 — Training the student

```bash
bash scripts/train_ner.sh {2000|5000|10000|full} [GPU_ID]
```

Runs `spacy debug config`, `spacy debug data`, then `spacy train` with
`configs/ner_sci.cfg` and `--code configs/custom_code.py`, writing to
`models/ner_<size>/`. GPU ID defaults to `-1` (CPU).

Two things in the config matter:

- **`components.tok2vec` is sourced from `en_core_sci_sm` and is deliberately
  *not* frozen**, so the pretrained feature extractor adapts to the corpus. The
  NER head is fresh, because the base model's generic `ENTITY` head is
  incompatible with the `DISEASE`/`MEDICATION`/`PROCEDURE` label set.
- **`configs/custom_code.py` registers `airms.scispacy_tokenizer.v1`**, an
  after-creation callback that installs scispaCy's tokenizer. `spacy train`
  otherwise builds a plain English tokenizer, and the mismatch does not raise —
  it silently degrades training, because entity boundaries falling inside a
  token cannot be learned. In `spacy debug data`, misaligned entity spans should
  be near zero; a large count means the callback did not fire, and you should
  not train until it does.

The same registration is why `src/ner/assertion.py` imports `custom_code.py` for
its side effects before any `spacy.load` — a plain load fails with
`RegistryError [E893]`.

Training params (in `configs/ner_sci.cfg`): dropout 0.1, patience 3500,
max_steps 60000, eval every 500, seed 7, Adam at 1e-3.

`DEV_DATA` defaults to `splits/dev_500.spacy` and is overridable by environment
variable. See [Reproducibility](#what-can-and-cannot-be-reproduced) — this file
cannot be regenerated.

### 6 — Evaluation

```bash
python eval_gold.py    --gold <gold.spacy> --models models/ner_* --split
python eval_teacher.py --gold <gold.spacy>
```

Both report exact-match and partial-overlap precision/recall/F1 per label and
overall. Partial matching is a deterministic greedy one-to-one assignment,
sorted longest-span-first, because boundary disagreement and missed entities are
different failure modes.

`--split` reports **pilot** (6 notes, annotated with INCEpTION recommenders
active) and **extension** (10 notes, no machine assistance) separately. Those
two halves are not methodologically identical, and a large gap between them is a
confound rather than a finding.

`eval_teacher.py` measures the ceiling: the student can only learn what the
teacher proposed, so if teacher recall against gold is low, no amount of silver
data fixes it.

Results as of 2026-08-30 are captured verbatim in
[`docs/eval_2026-08-30_models.txt`](docs/eval_2026-08-30_models.txt) and
[`docs/eval_2026-08-30_teacher.txt`](docs/eval_2026-08-30_teacher.txt). Gold set:
16 documents, 1,647 entities.

### 7 — The assertion layer

`src/ner/assertion.py` loads a trained model and appends medspaCy ConText.
`build_assertion_pipeline()` is the single entry point used by every downstream
consumer.

Structure of the pipeline it builds, and why:

- **PyRuSH is inserted after NER**, not before. The trained models contain no
  parser or sentence segmenter, and ConText needs sentence boundaries; inserting
  a component before NER would risk perturbing the entity predictions the model
  was evaluated on. PyRuSH rather than spaCy's punctuation `sentencizer`,
  because problem lists and headers have no terminal punctuation and a whole
  section would otherwise collapse into one sentence.
- **The sectionizer is capped at `max_section_length=120`** and its section
  category is descriptive only. The AIR·MS export has no line breaks, so a
  section runs to the next *matched* header; capping bounds the leak without
  discarding the signal that disabling section attributes entirely would cost.
- **`src/ner/allergy.py` runs last**, setting `ent._.is_allergy` from two
  mechanisms: header regions (which ConText cannot reach, being
  sentence-bounded) and inline cues (which are ConText's job). A drug in an
  allergy list is a drug the patient did *not* receive.
- Two packaged medspaCy rules are constrained in place — `prophylaxis` is
  restricted to DISEASE targets and `: no` is capped at `max_scope=5`. See
  `PACKAGED_RULE_FIXES` and `archive/patch_assertion_packaged_rules.py`.

Scoring against the gold attributes, with gold spans injected directly as
`doc.ents` so the assertion components are measured in isolation from NER
recall:

```bash
python score_assertions.py --export <unzipped export dir> \
    --model models/ner_full/model-best --errors-csv <out.csv>
```

Review tooling is documented in [`docs/assertion.md`](docs/assertion.md).

### 8 — Feature matrix

```bash
# note-level counts (runs inference; slow)
python build_feature_matrix.py --out outputs/feature_matrix_note.csv

# roll up without re-running inference
python build_feature_matrix.py --from-counts outputs/feature_matrix_note.csv \
    --level visit --out outputs/feature_matrix_visit.csv
```

Runs the assertion pipeline over the corpus, gates every entity, matches
survivors against `lexicon.yaml`, and emits counts at note level.

**Four axes gate.** An entity failing any of them is not evidence the thing
occurred: `is_negated`, `is_family`, `is_hypothetical`, `is_allergy`.

**Two axes deliberately do not gate.** `is_uncertain` is not a gate — "possible
pneumonia" is weak evidence, but it is evidence, and these features are sparse;
uncertain mentions are counted separately so the sensitivity can be reported.
`is_historical` is not a gate either, because temporality scores 60.7 F1 and
gating on it would silently zero out most true positives; instead each feature
carries a companion count of historical mentions, from which an `_all_hist` flag
is derived. Report it, do not enforce it.

The intermediate CSV stores four counts per feature (total, historical,
uncertain, dropped) rather than booleans, because counts roll up by summation to
any level and booleans do not. `--level` derives the binary view.

**On counts:** documentation volume differs sharply between label groups in this
cohort (cases ~13 notes/visit, controls ~2.7), so raw counts partly measure how
much was written. Prefer the binary features, or normalise.

The lexicon curation loop that produced `lexicon.yaml`:

```bash
python review_lexicon.py --report coverage
python review_lexicon.py --report terms --feature dialysis_access
python review_lexicon.py --report unmapped --top 60
```

The corpus decides the vocabulary, not memory — it contains surface forms nobody
predicts (`av fistula` 274, `avf` 211, `foley` 608) and traps that only show up
in review (bare `graft` matched 400 occurrences of coronary graft and
graft-versus-host disease).

---

## What can and cannot be reproduced

### Cannot be reproduced outside the enclave

Everything below is PHI or derived from it, is gitignored, and stays on the
enclave. An outside reader can read the code and the aggregate reports, not
rebuild the results.

| Artefact | Why |
|---|---|
| `cohort_all_notes.parquet` and all note text | PHI. The source cohort lives in the AIR·MS HANA warehouse behind an SSH tunnel. |
| `annotations/*/verified/*.json` | Contain full note text. |
| `annotations/gold_export/gold.spacy`, `gold_spans.csv` | Contain note text. |
| `splits/*.spacy`, `splits/manifest.csv`, `gold_candidates.csv` | Note text and patient identifiers. |
| `models/ner_*` | Trained on PHI; large binaries. |
| `outputs/`, feature matrices | Contain note and patient identifiers. |
| The Ollama teacher endpoint | Enclave-internal, token-authenticated. |
| The INCEpTION server | Enclave-internal. |

### Cannot be reproduced even inside the enclave

- **`splits/dev_500.spacy`.** All four training runs used it —
  `--paths.dev splits/dev_500.spacy` appears in every training log. It is a
  random 500-document subset of `dev.spacy` created on 13 August by a command
  that was not saved, and seeds 0, 7 and 42 under `numpy` and `random` do not
  reproduce it. The file is retained on the enclave. Treat it as a fixed
  artefact of the experiment, not as a regenerable intermediate: re-drawing a
  different 500 documents would make the four model results incomparable with
  the reported ones.

### Can be reproduced from this repository alone

- The full pipeline logic, every rule, and every threshold.
- `docs/annotation_guidelines.md` — the annotation standard.
- `lexicon.yaml` — 48 features with per-feature provenance notes.
- `configs/ner_sci.cfg` — the exact training configuration.
- `splits/split_report.md` — aggregate split counts, seed, tokenizer.
- `docs/eval_2026-08-30_*.txt` — the reported evaluation numbers.
- The offline smoke test: `python -m src.cli preannotate --synthetic-fixtures`
  and `python -m src.cli mock-e2e`, both of which use synthetic notes and touch
  no real data source.
- `pytest tests/` — covers span verification, the serializers, and patient
  extraction.

---

## Environment

`requirements.txt` pins only `medspacy==1.3.1`. There is no complete environment
specification in the repository; the following are what the code requires and
what the artefacts were produced with.

| Component | Value | Evidenced by |
|---|---|---|
| spaCy | 3.x, config schema v3 | `configs/ner_sci.cfg` |
| Base model | `en_core_sci_sm` (scispaCy) | `configs/custom_code.py`, `build_splits.py`, `inception_to_docbin.py` |
| Assertion | `medspacy==1.3.1`, ConText + PyRuSH + sectionizer | `requirements.txt`, `src/ner/assertion.py` |
| Teacher | `llama3.3:70b` via a local Ollama endpoint; see known gap 4 on how this is evidenced | `src/ner/preannotate.py`, `docs/annotation_guidelines.md` |
| Others | `pandas`, `numpy`, `pyyaml`, `typer`, `rich`, `loguru`, `requests`, `python-dotenv` | imports across `src/` and the root scripts |

Verify the tok2vec width before training in a new environment:

```bash
python -c "import spacy; print(spacy.load('en_core_sci_sm').config['components']['tok2vec']['model']['encode']['width'])"
```

It must match `components.ner.model.tok2vec.width` in `configs/ner_sci.cfg`
(currently 96).

---

## PHI handling

`.gitignore` blocks, path-independently: `outputs/`, `annotations/preannotations*/`,
`logs/`, and every `*.csv`, `*.tsv`, `*.parquet`, `*.spacy` and `*.zip`. Do not
relax these. Several scripts print an explicit reminder when they write a file
that contains note text.

Scripts that must run on a compute node inside the enclave, because they read
note text: `build_feature_matrix.py`, `score_assertions.py`,
`measure_allergy_sections.py`, `review_lexicon.py`, `rewrite_tsv_layer.py`,
`import_preannotations.py`, `sample_allergy.py`, `project6_to_spans.py`.

`scripts/assertion_report.py` and `scripts/assertion_sample.py` refuse to write
outside `outputs/` and `annotations/`.

---

## Known gaps

Recorded rather than fixed, because several are facts about how the work was
actually run.

1. **`splits/dev_500.spacy` is not regenerable.** See above. This is the largest
   reproducibility gap.
2. **Default model path differs between scripts.** `build_feature_matrix.py`,
   `score_assertions.py` and `sample_allergy.py` default to
   `models/ner_full/model-best`; `scripts/assertion_report.py` and
   `scripts/ablate_context_rules.py` default to `models/ner_10000/model-best`.
   Pass `--model` explicitly.
3. **Gold DocBin path is inconsistent.** Most scripts default to
   `annotations/gold_export/gold.spacy`; the usage examples in `eval_gold.py`
   and `eval_teacher.py` show `/tmp/gold16.spacy`.
4. **The production teacher model is not recorded in the run artefacts.** The
   model came from `$OLLAMA_MODEL` at run time, and
   `annotations/batch01_v2`, `batch02` and `batch03` carry only verification
   counters -- no config snapshot and no job log survives. The identification as
   `llama3.3:70b` rests on the run-directory naming
   (`preannotations_c1_llama33`, distinct from the `gemma3` and
   `v0325_llama70b` comparison runs) and on
   `docs/annotation_guidelines.md`, which names it. Note that
   `archive/compare_runs.py` names `llama3.1:70b`: that script belongs to the
   earlier llama-versus-gemma comparison and does not describe the production
   run. Future runs should snapshot the resolved model name alongside
   `run_summary.json`.
5. **`docs/assertion.md` says the report writes "four flags"**; it writes five.
   `is_uncertain` was added later and is missing from that document's attribute
   list.
6. **Two files named `annotation_guidelines.md`.** `docs/annotation_guidelines.md`
   is the hand-written thesis standard. `src/ner/annotation_schema.py`
   *generates* a different, much shorter one at
   `annotations/annotation_guidelines.md`. Only the first was used.
7. **`lexicon.yaml`'s header says 47 features**; it defines 48. Feature 48,
   `other_indwelling_device`, is corpus-derived rather than literature-derived
   and its provenance differs from features 1–47 — the file says so in place.
8. **Scripts with unresolvable inputs.** `which_cue.py` and `which_cue_exp.py`
   read `/tmp/gold16_txt/`; `archive/check_recommender_overlap.py` reads
   `/tmp/gold_verify_spans.csv` and a `gold25_backup_*.zip`; `review_lexicon.py`
   needs `lexicon_candidates.csv`. None of these is produced by anything in the
   repository; all were enclave-local scratch files.
9. **`split_docbin.py`** appears superseded by `build_splits.py` — it is a
   two-way split of a single `contract_docbin` DocBin, keyed on `patient_id`,
   which only `preannotation_serializers.write_contract_docbin` writes. Kept
   because that cannot be confirmed from the code alone.
10. **Duplicated scoring logic.** `eval_gold.py` and `eval_teacher.py` carry
    identical `prf`/`overlaps`/`score`/`report` blocks; `teacher_vs_gold.py` and
    `analyse_teacher_misses.py` re-implement `prf` again.
    `build_feature_matrix.py` and `review_lexicon.py` each implement `__ref__`
    expansion for the lexicon. Left alone: `eval_gold.py` and `eval_teacher.py`
    produced committed results and should not be perturbed.
11. **No test coverage** for the assertion layer or the feature matrix.

## Inherited scaffold

`src/cli.py` and the modules beneath it are a Track A / Track B design that
predates this work. Four of them — `src/cohort/cohort_builder.py`,
`src/preprocessing/note_preprocessor.py`, `src/features/feature_aggregator.py`,
`src/evaluation/evaluator.py` — are **specification only**: every method body is
`pass` beneath a docstring describing intended behaviour. The wrappers that
drive them (`scripts/run_cohort_builder.sh`, `run_preprocessing.sh`,
`run_feature_extraction.sh`, `run_evaluation.sh`, `start_airms_tunnel.sh`) run to
completion and do nothing. `python -m src.cli build-cohort` logs
`Cohort built: 0 persons` and exits 0. **Do not use them.**

`src/ner/model_trainer.py` and `src/ner/ner_extractor.py` are implemented for
the spaCy track and raise `NotImplementedError("out of thesis scope")` for the
HuggingFace track. They are superseded by `scripts/train_ner.sh` and
`build_feature_matrix.py` respectively, and are retained because the `mock-e2e`
plumbing check uses the trainer.

None of this can be deleted: `src/cli.py` imports all fourteen `src/` modules at
module load, and two of its commands — `preannotate` and `extract-patient` — are
live pipeline stages.

`docs/inventory.md` has the file-by-file breakdown.
