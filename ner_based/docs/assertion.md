# Clinical assertion review

The assertion layer runs after the fine-tuned NER model. It does not change
gold `Doc.ents`: the annotation contract remains entity span plus one of
`DISEASE`, `MEDICATION`, or `PROCEDURE`.

Run these commands from `ner_based/`:

```bash
python scripts/assertion_report.py
python scripts/assertion_sample.py
```

The report defaults to `annotations/gold_export/gold.spacy`,
`models/ner_10000/model-best`, and
`outputs/assertion_review/assertion_report.csv`. To review another learning
curve model or write a named report:

```bash
python scripts/assertion_report.py \
  --model-path models/ner_5000/model-best \
  --output outputs/assertion_review/ner_5000.csv
```

It reruns every note through NER and medspaCy ConText, writes the predicted
entity, offsets, four flags, and up to 60 characters of context on each side,
then prints flag counts and within-group percentages overall and for each NER
label. Flags are not mutually exclusive.

The sampler takes predicted-positive entities for each flag, spreads choices
across notes before taking a second entity from the same note, and writes a
blank `correct` column. The default is at most 50 rows per flag:

```bash
python scripts/assertion_sample.py \
  --input outputs/assertion_review/assertion_report.csv \
  --output outputs/assertion_review/assertion_sample.csv \
  --per-class 50 \
  --seed 7
```

An entity with multiple flags can appear in multiple `review_class` strata.
Mark `correct` consistently (for example, `1` or `0`) and calculate precision
within each `review_class`. This predicted-positive design estimates precision,
not recall. Both scripts refuse to write CSV files outside `outputs/` or
`annotations/`; those directories are already gitignored because the excerpts
contain PHI.

## Attributes

- `is_negated`: the entity is explicitly absent (for example, “no pneumonia”).
- `is_historical`: the entity is presented as prior history or status post.
- `is_hypothetical`: the entity is conditional, planned, possible, or part of
  a differential rather than established.
- `is_family`: the entity refers to a family member or family history rather
  than the patient.

A `False` value means no matching ConText modifier was applied; it is not proof
that the mention is definitively current, affirmed, and patient-experienced.

## Known limitations

Rule-based assertion detection is sensitive to wording and scope. Coordination
can attach a modifier to the wrong entity or stop it too early. Lexically fused
negations such as `non-tender` do not contain a standalone cue and may be
missed. Prevention language also needs semantic interpretation: in `DVT PPx`,
the condition is being prevented rather than asserted, but the shorthand does
not directly express a standard ConText status.

Some AIR.MS rules are deliberately review targets. `r/o` is mapped to negation
as specified even though it often means an unresolved “rule out”; the expanded
`rule out` is mapped to hypothetical. `versus` and `vs` use bidirectional scope
so both differential diagnoses are marked hypothetical. Broad cues such as
`if`, `should`, `not`, relationship words, and section abbreviations can produce
false positives. Because section headings are not standardized, any line-break
token terminates scope; this prevents leakage into the next section but can
truncate a modifier across a manually wrapped line. `but`, `however`, and `;`
also terminate scope.

There are no gold assertion labels yet. All assertion figures produced from
these CSVs are review-based estimates, not corpus-level gold evaluation.
