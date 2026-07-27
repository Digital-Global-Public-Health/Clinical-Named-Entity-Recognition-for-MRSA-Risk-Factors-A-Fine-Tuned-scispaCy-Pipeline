# NER Training Data Contract

This file is the source of truth for Track A training and evaluation data.

## Unit Of Annotation

- One spaCy `Doc` is one clinical note.
- Do not split one note into sentence-level training examples.
- Each `Doc` must carry:
  - `doc.user_data["note_id"]`
  - `doc.user_data["patient_id"]`
- Splits must be patient-level: no `patient_id` may appear in more than one of train, dev, or test.

## Entity Labels

Only these labels are in scope for Track A:

- `DISEASE`
- `MEDICATION`
- `PROCEDURE`

Gold annotation currently consists of spans and labels only. Gold entity spans live in `doc.ents`. Do not add assertion, temporality, experiencer, severity, negation, or other attributes to `doc.ents`.

## Span Construction

All spans must be created from character offsets with:

```python
span = doc.char_span(start_char, end_char, label=label, alignment_mode="strict")
```

If `doc.char_span(...)` returns `None`, the failed span must be logged and counted. A failed alignment must never be silently dropped.

The required failure log fields are:

- `note_id`
- `patient_id`
- `start_char`
- `end_char`
- `label`
- the intended span text, when available

## Attribute Sidecar

The attribute sidecar is reserved for a later phase. Assertion, temporality,
and experiencer are not part of gold annotation for now; medspaCy ConText will
handle these attributes later at inference time.

During the transition, a compatibility CSV sidecar may be written because some
existing training plumbing expects a sidecar file keyed by exact span
coordinates:

```text
(note_id, start_char, end_char) -> assertion, temporality, experiencer
```

Compatibility sidecar columns:

- `note_id`
- `patient_id`
- `start_char`
- `end_char`
- `label`
- `text`
- `assertion`
- `temporality`
- `experiencer`

Reserved attribute fields must be left empty unless/until the later attribute
phase is explicitly implemented. Do not map LLM context guesses into these
columns.

If the attribute layer is reintroduced, the vocabulary must match the
annotation guideline:

- `assertion`: `PRESENT`, `NEGATED`, `POSSIBLE`, `ALLERGY`
- `temporality`: `CURRENT`, `HISTORICAL`, `HYPOTHETICAL`
- `experiencer`: `PATIENT`, `OTHER`

## Expected Files

Training/evaluation uses spaCy DocBins:

- `annotations/train.spacy`
- `annotations/val.spacy` or `annotations/dev.spacy`
- `annotations/test.spacy`

Compatibility sidecars, when written, should be split the same way:

- `annotations/train_attributes.csv`
- `annotations/val_attributes.csv` or `annotations/dev_attributes.csv`
- `annotations/test_attributes.csv`

Mock data may live under `annotations/mock/` with the same filenames.

## Explicit Non-Goals

The sidecar is reserved and should not be used as gold metadata at this stage.
ConText, assertion resolution, temporality resolution, experiencer resolution,
and index-date leakage filtering are blocked until the supervisor defines the
index-event semantics.
