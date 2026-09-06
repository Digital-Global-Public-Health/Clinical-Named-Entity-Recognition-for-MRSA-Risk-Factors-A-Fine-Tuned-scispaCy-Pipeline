# archive/

Scripts that have already been applied. They are history, not build steps:
running them again would either re-apply an edit that is already in the source
or re-run an experiment whose outcome the pipeline now encodes.

Nothing in the live pipeline imports anything here.

**Run from `ner_based/`, not from this directory.** Every script resolves its
paths relative to the working directory (`lexicon.yaml`, `splits/manifest.csv`,
`annotations/...`), so moving the files did not change how they are invoked:

```bash
python archive/patch_lexicon_01.py    # from ner_based/
```

The `Usage:` lines inside the scripts themselves still show the old root-level
invocation (`python patch_lexicon_01.py`). Those docstrings are left as written
because several record the reasoning behind the change; prefix the path with
`archive/` when running one.

## Lexicon and rule patches

Each rewrites a source file in place, aborting if its anchor text does not match
exactly once. Their effects are already present in the files they target, so
re-running them will abort rather than double-apply.

| Script | Target | Change |
|---|---|---|
| `patch_lexicon_01.py` | `lexicon.yaml` | Five fixes from the first corpus review: drop the CAD patterns from `myocardial_infarction`, tighten `surgery` (bare `otomy`, `postoperative`, bare `or`), add `dlbcl` and `gvhd`, drop bare `ALL` |
| `patch_lexicon_02.py` | `lexicon.yaml` | Seven fixes from the second review: exclude `hd-mtx` and midline catheters, remove CPAP/BiPAP from `supplemental_oxygen`, add `pna`, `vanc`, `olt`, `ddkt`, `cabg`, `pci` |
| `patch_lexicon_02b.py` | `lexicon.yaml` | Allow PROCEDURE spans to match `immunosuppressed_state` |
| `patch_lexicon_03.py` | `lexicon.yaml` | Added feature 48 `other_indwelling_device`; moved `oht` to `immunosuppressed_state`; acronym sweep on rows 120-300 |
| `patch_assertion_packaged_rules.py` | `src/ner/assertion.py` | Constrained two packaged medspaCy ConText rules that mis-fire on this corpus: `prophylaxis` restricted to DISEASE targets, `: no` capped at `max_scope=5`. Now `PACKAGED_RULE_FIXES` in that module |

## Data migration

| Script | What it did |
|---|---|
| `reverify.py` | Re-derived span offsets for `annotations/batch01/` by replaying the stored proposals through the corrected word-boundary matcher, without calling the LLM. Model output held constant; only offset logic changed. Produced `annotations/batch01_v2/`, which is what `build_splits.py` reads |

## Teacher model selection

| Script | What it decided |
|---|---|
| `compare_runs.py` | Per-note span overlap and Jaccard between `llama3.1:70b` and `gemma3:27b` pre-annotations |
| `procedure_gap.py` | The same comparison restricted to PROCEDURE spans, the label the two models disagreed on most |
| `test_prompt_variants.py` | Whether the teacher's recall improves more from reframing the prompt as exhaustive extraction or from multi-pass union. Settled the production prompt now in `src/ner/preannotate.py`. Note the `test_` prefix is historical and does not indicate a pytest module; it lives here partly so `pytest` no longer collects it |

Outcome: llama was kept as the teacher, with the current single-pass prompt.

## Gold set extension (6 notes -> 16)

| Script | What it decided |
|---|---|
| `check_reuse.py` | Which of the six already-annotated pilot notes could be reused, and which test-split patients carry all four note types |
| `draw_new_gold.py` | Drew the ten extension notes (seed 7, patients P02/P03/P08/P09 x four note types) and wrote `splits/new_gold_10.csv` |
| `check_recommender_overlap.py` | How many gold spans originated as accepted INCEpTION recommender suggestions. This is why `eval_gold.py --split` reports pilot (recommenders active) and extension (no assistance) separately |

`check_recommender_overlap.py` reads `/tmp/gold_verify_spans.csv` and a
`gold25_backup_*.zip` in the working directory. Neither is produced by anything
in this repository; both were enclave-local at the time it was run.
