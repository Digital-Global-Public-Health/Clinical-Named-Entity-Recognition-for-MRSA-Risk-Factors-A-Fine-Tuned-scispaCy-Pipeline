# Clinical assertion layer

The assertion layer runs after the fine-tuned NER model and adds attributes to
predicted entities. It never changes spans or labels: the annotation contract
remains entity span plus one of `DISEASE`, `MEDICATION`, `PROCEDURE`.

## Components

Four spaCy components, in fixed order after `ner`:

1. `medspacy_pyrush` — sentence segmentation. Used rather than spaCy's
   punctuation sentencizer because header and list text carries no terminal
   punctuation.
2. `medspacy_sectionizer` — assigns each entity a section category, and writes
   assertion attributes for entities inside sections such as *family history*
   and *past medical history*. Bounded by `max_section_length=120` tokens.
3. `medspacy_context` — 185 ConText rules applied within a sentence.
4. `airms_allergy` — flags MEDICATION spans named as allergens rather than
   administered, by inline cue or by header region.

An assertion therefore has two possible sources: a ConText cue near the entity,
or the section the entity sits in. The section path is why
`max_section_length` matters — see the limitations below.

## Axes

| attribute | ConText category | set when |
|---|---|---|
| `is_negated` | `NEGATED_EXISTENCE` | the note asserts the finding is absent |
| `is_uncertain` | `POSSIBLE_EXISTENCE` | suspected, or part of a differential |
| `is_hypothetical` | `HYPOTHETICAL` | conditional, planned, or future-framed |
| `is_historical` | `HISTORICAL` | the note asserts the mention predates this episode |
| `is_family` | `FAMILY` | the mention belongs to a relative |
| `is_allergy` | — | MEDICATION named as an allergen (`airms_allergy`) |

All default to false. A false value means no matching modifier fired; it is not
positive evidence that the mention is current, affirmed and patient-experienced.

`airms_allergy` also writes `ent._.allergy_source`, recording which mechanism
fired: `header`, `cue`, or both joined with `+`. The two paths differ in
precision, so evaluation reports them separately.

Temporality is note-relative. The extract carries no date for either the note or
the outcome, so `is_historical` records what the text asserts rather than a
position relative to an index event.

## Corpus-specific rule changes

The export contains no newline characters, so the packaged newline terminator
can never fire. Structural terminators were substituted: a run of two or more
whitespace characters, and the bullet characters that separate problem-list
entries.

Four packaged rules were constrained. `as needed` and a bare `family` cue were
disabled, both contradicting the annotation guideline. A `prophylaxis` modifier
firing backward with no target restriction was limited to DISEASE targets, and a
`: no` modifier reaching across section boundaries was capped at five tokens.
See `archive/patch_assertion_packaged_rules.py`.

`r/o` was remapped from negation to `POSSIBLE_EXISTENCE`, matching `rule out`.
Past-tense `ruled out` remains negation. This diverges from the AIR.MS
specification and matches the annotation guideline instead.

The hypothetical axis was rebuilt: its packaged inventory held two cues, and
fourteen cue families were added. `pending` and `ordered` are declared backward,
because in this corpus they follow the entity, and are capped at five tokens.

Family history arrives as a table in which the relation word follows the
condition, so packaged forward-scoping family rules cannot reach the entity.
Backward-scoping rules for the grandparent relations were added with a scope of
five tokens.

## Scoring

The layer is scored against the gold assertion pass with gold spans injected
directly as document entities and only the assertion components run, so the
measurement isolates the rules from entity-recognition error. See
`score_assertions.py`.

## Known limitations

Rule-based assertion detection is sensitive to wording and scope. Coordination
can attach a modifier to the wrong entity or stop it too early, and lexically
fused negations such as `non-tender` contain no standalone cue.

Historicity is the weakest axis, and the reason is structural rather than a
matter of rule quality: what marks a mention as historical is tense, date and
position in a chronology, none of which a trigger-phrase matcher can see.
Accuracy is correspondingly note-type dependent, highest where a
past-medical-history section gives cues something to scope over and lowest in
discharge summaries.

Section boundaries leak. With no newlines a section runs to the next *matched*
header, so one unrecognised header lets a section absorb everything beneath it.
`max_section_length` bounds the damage rather than removing it.