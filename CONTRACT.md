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

Gold entity spans live in `doc.ents`. Do not add assertion, temporality, experiencer, severity, negation, or other attributes to `doc.ents`.

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

Entity attributes are stored outside the `.spacy` files in a CSV sidecar keyed by exact span coordinates:

```text
(note_id, start_char, end_char) -> assertion, temporality, experiencer
```

Required sidecar columns:

- `note_id`
- `patient_id`
- `start_char`
- `end_char`
- `label`
- `text`
- `assertion`
- `temporality`
- `experiencer`

Defaults when a field is absent upstream:

- `assertion=PRESENT`
- `temporality=CURRENT`
- `experiencer=PATIENT`

Current allowed values:

- `assertion`: `PRESENT`, `NEGATED`, `UNCERTAIN`
- `temporality`: `CURRENT`, `HISTORICAL`, `PLANNED`
- `experiencer`: `PATIENT`, `OTHER`

## Expected Files

Training/evaluation uses spaCy DocBins:

- `annotations/train.spacy`
- `annotations/val.spacy` or `annotations/dev.spacy`
- `annotations/test.spacy`

Matching sidecars should be split the same way:

- `annotations/train_attributes.csv`
- `annotations/val_attributes.csv` or `annotations/dev_attributes.csv`
- `annotations/test_attributes.csv`

Mock data may live under `annotations/mock/` with the same filenames.

## Explicit Non-Goals

The sidecar is only read as metadata at this stage. ConText, assertion resolution, temporality resolution, experiencer resolution, and index-date leakage filtering are blocked until the supervisor defines the index-event semantics.
