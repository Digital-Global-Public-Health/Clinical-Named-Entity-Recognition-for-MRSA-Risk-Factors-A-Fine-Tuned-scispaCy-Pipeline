# MRSA NLP — Annotation Guidelines (v3)

**Author:** Tobias Rademacher
**Status:** active — use this version for the gold set and its extension
**Supersedes:** v2 (`cc89c64`, 2026-08-19)

**Changes from v2:** non-invasive imaging is now PROCEDURE (§2.3); assertion
attributes added as §3A; the anatomy connector test added to §4; ~25 rules
settled during the gold pilot and the assertion pass folded into §5; §1, §6 and
§7 amended. Markdown headings restored. All entity-level decisions previously
marked "decision needed" (D3, D4, D5, `MSMS`) are closed — the only open items
in §7 are the imaging sign-off, PROCEDURE recall, and the IAA subset.

---

## 1. Purpose

These guidelines define how to annotate clinical notes from AIR·MS for training
and evaluating a named entity recognition (NER) model. The model's goal is to
extract clinical signals relevant to MRSA bacteremia risk prediction.

Two distinct uses:

- **Silver corpus** — `llama3.3:70b` pre-annotates notes using a compressed form
  of these rules (`src/ner/preannotate.py`); a human corrects them in INCEpTION.
- **Gold set** — held-out notes annotated by hand with **no pre-annotation**,
  used to measure how good the silver corpus and the student model actually are.
  Design: four test-split patients × four note types = 16 notes.

**Governing principle.** Entity annotation records span + label only. Whether a
mention is negated, historical, hypothetical, or about someone other than the
patient is *not* part of the entity decision: tag the span regardless of
context, and never tag the cue itself.

For the **gold set only**, a second annotation pass records those contextual
judgments as attributes on the already-annotated spans (§3A). This pass exists
to measure the downstream medspaCy ConText layer, which is otherwise
unvalidated. The silver corpus is not attribute-annotated; the teacher proposes
spans and labels only.

---

## 2. Entity types

Three labels. Anything that does not fit one of the three is left unlabeled.

### 2.1 DISEASE

Any clinical condition, diagnosis, symptom, syndrome, or infection.

**Includes**

- Named diseases: pneumonia, rheumatoid arthritis, MRSA bacteremia
- Symptoms: fever, hypotension, shortness of breath
- Conditions: immunosuppression, renal failure, diabetes mellitus
- Wounds and skin findings: decubitus ulcer, cellulitis, chronic wound
- Standard abbreviations: RA, CHF, DM2, UTI, COPD, ICH
- Prior infections: MRSA bacteremia in "history of MRSA bacteremia (2021)"
- Substance-use **disorder** terms and noun-phrase substance mentions:
  `crack use`, `Polysubstance Use`, `substance use disorder`

**Excludes**

- Body parts alone: left arm, L leg, LLE
- General descriptors: ill, sick, unwell, not doing well
- Lab and test names or values — see §2.4
- Consumption described as an activity with a quantity:
  `drinking 2-3 beers daily` → skip

### 2.2 MEDICATION

Any drug, drug class, or pharmacological agent named in the note.

**Includes**

- Drug names: vancomycin, prednisone, methotrexate, tacrolimus
- Drug classes: corticosteroids, antibiotics, immunosuppressants
- Abbreviations and short forms: abx, vanc, pip-tazo
- Allergens that are drugs (§3.4)
- **Drug classes named without a specific agent:** `antimicrobial`, `abx`.
  Route and the word `therapy` are stripped per §4 — `IV abx` → `abx`,
  `antimicrobial therapy` → `antimicrobial`.
- **Contrast agents and administered diagnostic substances:** `Gadobutrol` /
  `Gadavist`, `perflutren`, `sulfur hexafluoride`, `iodine`. These are
  pharmacological agents given to the patient and fall under this section by its
  own definition; downstream lexicon matching filters them out of the feature
  layer.

**Excludes**

- Dosage, strength, form, route, frequency — §4
- Vitamins and supplements unless clearly therapeutic
- **Non-pharmacological allergens.** Foods, environmental and material
  allergens are not pharmacological agents and fall outside §2.2 by
  definition: `shellfish`, `shrimp`, `latex`, `Inhalants`, `dust`. They are
  left unlabeled, so no allergy attribute applies (§3A.5).
- **Diluents.** `sodium chloride 0.9 %` as an IV vehicle → skip;
  `sodium chloride 1 gram tablet` → tag.

### 2.3 PROCEDURE

Any medical procedure, intervention, or indwelling device. Devices and lines are
procedures in this schema — this is the most-missed category, so read the list.

**Includes**

- Devices and lines: central line, PICC, tunneled cath, CVC, Foley, urinary
  catheter, tracheostomy
- Procedures: intubation, hemodialysis / HD, CABG, wound debridement, surgery,
  bone marrow transplant, lumbar puncture
- Abbreviations: BAV, RHC, CVVH, OHT, BMT
- **Non-invasive imaging: CT scan, chest X-ray, TTE, MRI, CTA.**
  ⚠️ **Schema change taken mid-pilot on 2026-08-14; awaiting supervisor
  sign-off.** v2 excluded these. Rationale: the invasive / non-invasive
  boundary carries almost no surface signal for a tok2vec model — `CTA` and
  `CVC` sit in near-identical contexts — and feature aggregation filters to
  specific master-list factors downstream, so extra imaging entities cost
  nothing there. All six pilot notes were re-annotated under the new rule;
  everything from note 7 onward follows it natively.
- Immunisation (see §5 D1)

**Excludes**

- Routine assessments: physical exam, vitals
- Care-coordination activity (see §5 D2)

### 2.4 Laboratory tests and results — NOT annotated

Lab names, panels and values are not entities in this schema. They are available
as structured EHR data and add nothing here.

- `WBC 12.3`, `blood cultures`, `CBC`, `creatinine 2.1`, `INR` → unlabeled
- But a diagnosis stated from a lab **is** DISEASE: `bacteremia`, `neutropenia`

---

## 3. Edge cases

### 3.1 Negation — annotate

Tag the entity even when explicitly negated. Do not tag the negation cue.

- "No pneumonia on imaging" → annotate `pneumonia`
- "Fever denied" → annotate `fever`
- "Ruled out for endocarditis" → annotate `endocarditis`

Rationale: negation is detected downstream by ConText, and recorded on the gold
set as `polarity` (§3A.1). Withholding the span would destroy the information
the assertion layer needs.

**Negated but named → tag.** `no LAD` → `LAD`; `without occlusion` →
`occlusion`; `negative for valvular vegetation` → `vegetation`.

**Normal descriptors with nothing named → skip.** `VF intact`, `MMM`, `CTAB`,
`supple`, `unremarkable`, `patent`.

**Negative-polarity functional deficits → skip.** `inability to abduct`,
`not able to visualize`, `non-tender`. There is no positive core to extract, and
a span containing the negation would be inverted by ConText. Positively-named
equivalents (`vision loss`, `CN VI palsy`) are tagged.

**Generic negatives → skip** (`no acute abnormality`, `no significant change`);
**specific negated findings → tag** (`no pneumothorax`, `no mass effect`).

**Slash-compressed multi-condition abbreviations → skip** (`no m/r/g`,
`no w/r/r`); tag when spelled out.

### 3.2 Abbreviations — annotate as the type they represent

`MRSA` → DISEASE · `PICC` → PROCEDURE · `RA` → DISEASE · `vanco` → MEDICATION

Where both an abbreviation and its expansion appear, tag them as two separate
spans — do not merge across the parenthesis (§4).

- "ICH (intracerebral hemorrhage)" → two DISEASE spans

If unsure what an abbreviation means, leave it unlabeled and record it in §5.

### 3.3 Speculation / uncertainty — annotate

Tag the entity; do not tag the hedge.

- "Rule out sepsis" → annotate `sepsis`
- "Possible pneumonia" → annotate `pneumonia`
- "Concern for line infection" → annotate `line infection`

### 3.4 Allergies — annotate the drug

- "Penicillin allergy" → annotate `Penicillin` as MEDICATION
- "Allergic to sulfa" → annotate `sulfa` as MEDICATION

Do not annotate the word "allergy" itself. Whether the drug was administered or
merely listed as an allergy is an assertion-layer distinction, recorded on the
gold set as `allergy` (§3A.5), not an entity one.

Non-pharmacological allergens are not annotated at all — see §2.2.

**Prophylaxis → skip the header, tag the drug.** `DVT PPx` names a condition
being *prevented*; tagging `DVT` would assert a diagnosis the patient does not
have, and ConText has no cue to catch it. Tag `HSQ` / `Lovenox` instead.

### 3.5 Historical mentions — annotate normally

- "History of MRSA bacteremia in 2021" → annotate `MRSA bacteremia`
- "s/p BAV (5/27/14)" → annotate `BAV` as PROCEDURE

Do not include the date in the span.

### 3.6 Family history — annotate

- "Mother had breast cancer" → annotate `breast cancer` as DISEASE
- "Father with MRSA wound infection" → annotate `MRSA wound infection`

Experiencer (patient vs. relative) is resolved by ConText downstream and
recorded on the gold set as `experiencer` (§3A.4).

### 3.7 Compound entities — one span per clinical concept

Tag the complete clinical term as a single span; do not split a term that names
one concept.

- `MRSA bacteremia` → one DISEASE span, not `MRSA` + `bacteremia`
- `central venous catheter` → one PROCEDURE span
- `acute kidney injury` → one DISEASE span

This is bounded by §4: the span is the clinical term itself, with decoration,
dosing and trailing anatomy excluded.

### 3.8 Coordinated entities — annotate separately

- "Fever and chills" → two DISEASE spans
- "Vanc + pip-tazo" → two MEDICATION spans

### 3.9 Repeated mentions — annotate every occurrence

If `LVAD` appears 44 times in a note, all 44 are annotated. Entity counts in this
project therefore report occurrences, not distinct entities.

### 3.10 Structured symptom scales

Tag every symptom name in an ESAS-style scale, **including items scored zero**.
The zero is an assertion of absence and is recorded on the polarity axis
(§5 C2), not by withholding the span.

### 3.11 Misspellings — tag verbatim

Annotate the surface form as written; never correct it. See §5 F for the
observed inventory.

---

## 3A. Assertion attributes (gold set only)

Annotated as features on the span layer in INCEpTION, one pass over the spans
already annotated under §2–§4. Five axes, each with a default. **Most spans are
all-default** — only exceptions are changed.

| feature | values | default |
|---|---|---|
| `polarity` | affirmed · negated | affirmed |
| `certainty` | certain · uncertain · hypothetical | certain |
| `temporality` | current · historical | current |
| `experiencer` | patient · other | patient |
| `allergy` | no · yes | no |

The axes are independent. A span may be negated *and* historical
(`no prior MRSA`), or allergic *and* negated (`denies penicillin allergy`).

### 3A.1 polarity — is the entity asserted to be absent?

`negated` when the note states the finding is not present. Follows §3.1: the
span was annotated anyway, and this is where the absence is recorded.

- `no pneumonia on imaging` → pneumonia, **negated**
- `fever denied` → fever, **negated**
- `negative for valvular vegetation` → vegetation, **negated**
- `ruled out for endocarditis` → endocarditis, **negated** (past tense = resolved)

Not negated: a normal or reassuring finding that names something present
(`wound clean and dry` → wound is affirmed).

### 3A.2 certainty — is the entity asserted, suspected, or conditional?

Two non-default values, and the distinction is **suspected now** vs
**conditional or future**.

`uncertain` — the entity may be present but is unconfirmed at the time of
writing:

- `possible pneumonia`, `concern for sepsis`, `c/f line infection`
- `may represent abscess`
- `pneumonia vs atelectasis` → **both** spans uncertain (differential)
- `rule out sepsis` → sepsis, **uncertain** (pending instruction, not a finding)
- **`r/o sepsis` → uncertain.** Decision: `r/o` is treated as the pending sense,
  matching `rule out`. The AIR·MS specification maps it to negation; this
  guideline diverges deliberately, because in this corpus the abbreviation
  overwhelmingly introduces an unresolved differential. The ConText rule was
  changed to match (`POSSIBLE_EXISTENCE`). Contrast §3A.1: *`ruled out`*, past
  tense, remains negation.

`hypothetical` — the entity is conditional, planned-against, or future-framed
and is not claimed to exist at all:

- `if she develops fever, start vancomycin` → fever, **hypothetical**
- `should he spike, obtain cultures` → spike, **hypothetical**
- `return if wound worsens` → wound worsening, **hypothetical**

Prophylaxis: per §3.4 the condition being prevented is not annotated as a span
at all (`DVT PPx` → tag `Lovenox`, not `DVT`), so it does not reach this axis.

### 3A.3 temporality — is the entity current or prior?

**Note-relative, not index-date-relative.** There is no index date in the cohort
export, so `historical` means *the note asserts this predates the current
episode*, judged from the text alone. This is a documented limitation: it is
weaker than the temporal separation the prior-MRSA feature (master list row 1)
ultimately requires.

`historical` when the text explicitly marks it as past:

- `h/o MRSA bacteremia`, `hx of C. diff`, `history of endocarditis`
- `s/p BAV (5/27/14)`, `status post CABG`
- `prior line infection`, `previous admission for cellulitis`
- an explicit past date attached to the mention
- `resolved`, `since resolved`, `treated in 2019`

`current` otherwise — including everything in the active problem list, the
assessment, and today's medications. **Absence of a history cue means current.**
Do not infer historicity from clinical plausibility; annotate what the text says.

Entities under a *Past Medical History* header are historical only if the header
is the sole cue — record the judgment in §5 either way, because section-derived
temporality is exactly what the sectionizer cannot reliably deliver on this
corpus.

See §5 A for the eight temporality rules settled during the pass.

### 3A.4 experiencer — is the entity about the patient?

`other` when the mention belongs to a relative or another person. Follows §3.6:
the span is annotated, and this is where the experiencer is recorded.

- `mother had breast cancer` → breast cancer, **other**
- `father with MRSA wound infection` → MRSA wound infection, **other**
- `FHx significant for diabetes` → diabetes, **other**

**Not `other`:** the bare word *Family* in a provenance or attribution line
(`Source: Patient, Family and Team`) does not make the surrounding findings
family-experienced. Those spans stay `patient`. This is a known false-positive
source in the current cue set.

### 3A.5 allergy — is a medication named as an allergen rather than given?

MEDICATION spans only. `yes` when the drug is named as something the patient
reacts to, not something administered. Follows §3.4: the drug is annotated as
MEDICATION, and this is where the distinction is recorded.

Two shapes:

- **inline cue** — `allergic to vancomycin`, `PCN allergy`, `sulfa intolerance`
- **allergy list / header region** — the drug appears under an `Allergies:` or
  `Allergen Reactions` header, with no local cue anywhere near it

Not `yes`: `allergic rhinitis`, `allergy testing`, `seasonal allergies` — these
are diseases or procedures, not drug allergies, and the drug axis does not apply.
Non-pharmacological allergens are not annotated at all (§2.2), so they never
reach this axis.

**Interaction with polarity.** `denies penicillin allergy` is `allergy=yes` *and*
`polarity=negated`: the note asserts the allergy does not exist. Both are
recorded; downstream, neither reading supports "the drug was administered."

**Known limitation.** Allergy status is a property of the *patient*, not the
note, so a gold set of four patients bounds this axis at four instances
regardless of how many notes are annotated. The allergy component is therefore
evaluated separately on a corpus sample, not on the gold set.

### 3A.6 What is not recorded

- Severity or acuity (`severe`, `acute`) — stripped from spans per §4, not an axis
- Certainty *about the annotation* — that belongs in the §5 questions log
- Section membership — descriptive only; see §6

### 3A.7 Procedure

1. The gold spans already exist in INCEpTION. This pass adds attributes to them;
   it does not re-annotate spans.
2. Values are constrained tagsets, not free text.
3. Every judgment call goes in the §5 questions log as it happens.
4. If an IAA subset is annotated, the second annotator works against this
   section independently, with no visibility of the first annotator's
   attributes.

---

## 4. Span boundary rules

**Principle:** tag the shortest span that carries the full clinical meaning.

**Always exclude**

- Articles and determiners: "a central line" → `central line`
- Severity and acuity modifiers: "severe pneumonia" → `pneumonia`; also
  `acute`, `mild`, `Stable`
- Trailing punctuation
- Leading bullets, numbering, and section labels: "• Hyperlipidemia" →
  `Hyperlipidemia`

**Always keep:** polarity modifiers that constitute the finding —
`abnormal intracranial enhancement`, `increased work of breathing`.

**MEDICATION — drug name only.** Exclude parenthetical brand names, dose,
strength, form, route and frequency.

- "omeprazole (PRILOSEC) 20 mg capsule Take 20 mg by mouth daily" → `omeprazole`
- "fluticasone-salmeterol (ADVAIR) 250-50 mcg" → `fluticasone-salmeterol`

**DISEASE / PROCEDURE — full clinical term**, minus trailing anatomical
qualifiers that are separate body-part mentions.

- "Macular degeneration of right eye" → `Macular degeneration`
- "Lumpectomy Right" → `Lumpectomy`
- "left thalamic ICH (2010)" → `ICH`

**The anatomy connector test.** Adjectival anatomy *inside* the term stays;
anatomy *introduced by a connector* is dropped.

- Keep: `abdominal pain`, `orbital cellulitis`, `medial rectus abscess`
- Drop after `of` / `in` / `at` / `involving`: `pain in his R eye` → `pain`;
  `abscess in R medial rectus muscle` → `abscess`
- **Laterality is always dropped**: `R`, `L`, `bilateral`

**Carve-out: the standalone-interpretability test.** The connector test strips
anatomy only when the remaining head noun is a self-sufficient clinical finding.
Ask: *would the stripped span be interpretable as a finding on its own?*

- Yes → strip. `pain in his R eye` → `pain`; `abscess in R medial rectus muscle`
  → `abscess`
- No → keep the anatomy. Bare morphological descriptors —`curvature`,
  `stenosis`, `atrophy`, `hypertrophy`, `dilatation`, `thickening`,
  `calcification` — name a shape, not a condition; the anatomy is what makes
  them clinical. `curvature of the nasal septum` → `curvature of the nasal
  septum`; `stenosis of the left renal artery` → `stenosis of the left renal
  artery`.

Laterality still drops in both cases.

**Abbreviation and expansion are separate spans.**

- "AS (aortic stenosis)" → `AS` and `aortic stenosis`, not one merged span

---

## 5. Questions log and settled rules

Record every judgment call made while annotating. This log is the source of
future guideline versions and is part of the assessed deliverable.

| Note ID | Excerpt | What I labeled | Why uncertain | Revisit? |
|---|---|---|---|---|
| | | | | |

The rules below were settled during the six-note gold pilot (2026-08-14) and the
gold assertion pass (2026-08-25/26). All are binding for the gold set extension.

### A. Temporality

**A1. History cues scope over a whole coordinated list.**
`PMHx of ADHD, bipolar disorder, crack cocaine use` → all three historical. The
cue appears once and governs every member of the list. Same for `hx of`,
`PMSH:`, `Past medical history includes`.

**A2. Four surface forms of the same cue.** `PMHx of`, `PMSH:`,
`Past Medical History:` (as header), `Past medical history includes`. Treat
identically. (Only `pmhx` is in `AIRMS_CONTEXT_RULES`.)

**A3. `s/p` inside the current encounter stays `current`.**
`s/p Piperacillin-Tazobactam (1/3-1/7)` during this admission → current.
`s/p BAV (5/27/14)` → historical. The cue marks a completed action; temporality
depends on whether that action falls inside the current episode.

**A4. A diagnosis with a past date but active treatment is `current`.**
`DM2 recently diagnosed (12/2021 A1c 10.1)` with insulin ordered → current. The
date attaches to the diagnosis *event*, not to the condition's persistence. The
rule is what the text asserts about the mention, not clinical truth.

**A5. A condition that is both the admission reason and noted as recurrent is
`current`.** The recurrence clause is context; it does not make the named entity
historical.

**A6. Presenting complaints marked `h/o … x 1 day` are `current`.**
The duration phrase places them in the current presentation.

**A7. Comparison studies in radiology reports are `historical`**, even when
dated one day earlier. Temporality is relative to the note, at whatever
granularity the text implies.

**A8. Narrative tense and bare dates are history cues for the annotator but not
for ConText.** `he returned 12/26`, `was put on`, `during this course`,
`CTA on 1/2/22`. Annotate historical; expect the pipeline to miss them.

### B. Certainty

**B1. An entity inside a differential is `uncertain` unless independently
corroborated elsewhere in the note.** Check for independent affirmation before
marking uncertain.

**B2. A hedge on a causal explanation propagates to the entities inside it.**
`Hyponatremia … Likely iso of SIADH` → SIADH uncertain.

**B3. Negative-polarity hedges are `uncertain`, not `negated`.**
`unlikely to be infective endocarditis`, `doubt`, `less likely`. Polarity
records what the note asserts is *absent*; "unlikely" stops short of assertion.

**B4. Monitoring lists are `hypothetical`.**
`Monitor for sedation, confusion, hallucinations, myoclonus` — side effects to
watch for, not present findings.

**B5. Planned or considered interventions are `hypothetical`.**
`planned for surgery on 1/12`, `may consider Haldol`, `Consider initiation of
liposomal amphotericin B`, `Suggest transfusion PRBC's`.

**B6. PRN indications are `affirmed`, not hypothetical.**
`as needed for Cough`, `as needed (constipation)`. The PRN framing describes when
to take the drug, not whether the condition exists. For some patients a
medication-list indication is the only place a symptom is documented; marking
these hypothetical would make them invisible to the feature layer.

**B7. `r/o` → uncertain**, matching `rule out`. Past-tense `ruled out` remains
negation. Divergence from the AIR·MS specification; the ConText rule was changed
to `POSSIBLE_EXISTENCE` to match.

### C. Polarity

**C1. Negation attached to *change* does not negate the entity.**
`no significant interval change in the right retrobulbar fluid collection` → the
collection is affirmed and present. Known ConText failure mode.

**C2. ESAS items scored `0-none` are `negated`.** The span is annotated per
§3.10; the zero score is an explicit assertion of absence.

**C3. `did not help` negates efficacy, not administration.**
`oxycodone did not help` → oxycodone affirmed; the drug was given.

**C4. A failed procedure is affirmed.**
`LP unsuccessful without return of CSF` → LP affirmed (the attempt occurred);
`CSF` negated if tagged.

### D. Entity-level rules

**D1. Immunisation is a PROCEDURE; the disease named as its target is not
annotated.** `fully vaccinated for COVID` → tag the vaccination, not COVID.
Follows the §3.4 prophylaxis logic: the prevented condition is not asserted.

**D2. Care-coordination activity is not a PROCEDURE.**
`SNF placement`, `SW consult`, `REAP appointment`, `pain management consult`.
SNF / long-term-care exposure *is* master-list row 23 — deliberately excluded
from NER as a structured feature, not an oversight. The Chapter 6 coverage
comparison must not count these as NER misses.

**D6. Held medications.** `[Held by provider] heparin`, `sodium bicarbonate
Held` — not administered, but no axis records this. They will read as
administered in the feature layer. Documented limitation.

**D7. `c/w` is ambiguous.** Before a treatment → "continue with" (affirmed
ongoing). Before a finding → "consistent with" (affirmed by inference). Both
affirmed, but the readings differ and a second annotator may split them.

**D8. Urology/exam shorthand.** `No CVAT`, `No D/C Noted` are specific negated
findings (→ tag, negate) but sit adjacent to compressed normal descriptors
`N/T`, `N/M`, `N/E` (→ skip). The boundary is thin; recorded as a known
inconsistency risk.

**D3. Drug classes named without an agent — tag.** *Closed 2026-08-27.*
`antimicrobial therapy`, `IV abx`. §2.2 already includes drug classes and
already lists `abx`; excluding these would contradict the section. Boundary
follows §4: route and `therapy` are decoration. That they are unmappable to a
master-list feature is not a reason to withhold the span — only ~31 % of entity
occurrences map to a feature at all, so unmappable is the normal case.

**D4. Contrast agents — tag.** *Closed 2026-08-27.* `Gadobutrol` / `Gadavist`,
perflutren (~124 spans), sulfur hexafluoride (64), iodine (104). These are
administered pharmacological agents and meet §2.2's definition; excluding them
would need a carve-out that cannot be stated without also excluding, say, a
sodium chloride tablet. The genuinely non-pharmacological material (latex,
shellfish, shrimp — ~390 spans) is handled by the §2.2 exclusion instead.

**D5. Bare morphological descriptors — anatomy retained.** *Closed 2026-08-27.*
See the standalone-interpretability test in §4. A model trained on `curvature`
as a DISEASE is learning noise; the extra annotator variance the carve-out
introduces is the cheaper cost.

### E. Abbreviations

- **`RHS` = right heart strain.** Context: `RHS on pocus` alongside PE concern.
  *Closed 2026-08-25.*
- **`H/N` = hydronephrosis.** Context: consult for elevated creatinine in a
  patient with `Ca-p` and an `indwelling ureteral stent`. *Closed 2026-08-25.*
- **`MSMS` — not tagged.** *Closed 2026-08-27.* Meaning never established;
  appears ~4× and consistently occupies the position of a facility or service
  name. If that reading is right it is out of scope anyway (facility and
  transfer exposure are master-list rows 22–25, excluded from NER as structured
  data). Do not tag it, and do not tag other unresolved acronyms that sit in
  facility/service position — record them here instead.

### F. Misspellings observed (tagged verbatim per §3.11)

Hydromorphone appears in **five** surface forms across the six pilot notes:
`dilaudid`, `Dilaudid`, `Dailudid`, `Diladudid`, `DILAUDID`. Also `zoysn`
(appearing in the same note as the correct `zosyn`), `opiods`, `subtherapuetic`,
`tachypnic`, `mrsa bactermia`.

A countable, reportable source of NER difficulty that more training data does
not fix. Extend this list as new forms appear.

---

## 6. Out of scope (deliberate, for the limitations section)

Recorded here so their absence is a documented choice rather than an oversight.

- **Assertion / temporality / experiencer.** Not part of *entity* annotation.
  Handled downstream by rule-based medspaCy ConText (cf. Harkema et al. 2009)
  plus an AIR·MS allergy component, and measured separately against gold
  attributes annotated per §3A. Temporality is note-relative, since the cohort
  export carries no index date.
- **Section membership.** The sectionizer assigns `ent._.section_category` as
  descriptive metadata only. Its boundaries leak — the AIR·MS export contains no
  newlines, so a section runs from its header to the next *matched* header.
  Nothing downstream reads it.
- **SEVERITY as a fourth label** (immunocompromised, critically ill) — optional
  in the exposé, dropped to keep the label set at three.
- **A SOCIAL label.** Would resolve the substance-use boundary in §2.1 more
  cleanly than folding disorder terms into DISEASE. Not adopted; a fourth label
  is a schema change, and PROCEDURE is already the thin one.
- **Entity linking to SNOMED CT / RxNorm** (exposé Phase 4).
- **Healthcare-context factors** — prior hospitalization, nursing-home exposure,
  facility transfer. Reliably coded in structured EHR fields; excluded from NER
  per the risk-factor master list (rows 22–25).

---

## 7. Open questions for supervision

Closed since v2:

- ~~Family history (§3.6)~~ — resolved: annotated, experiencer recorded (§3A.4);
  risk relevance is a feature-layer decision rather than an annotation one.
- ~~Allergies (§3.4)~~ — resolved: MEDICATION span, `allergy=yes`, excluded from
  administered-drug features via `is_administered()`. Non-pharmacological
  allergens excluded from the label entirely (§2.2).
- ~~D3 / D4 / D5~~ — resolved 2026-08-27: drug classes tagged, contrast agents
  tagged, anatomy retained for bare morphological descriptors. See §5 and §4.
- ~~`MSMS`~~ — resolved 2026-08-27: not tagged (§5 E).

Open:

- **Imaging → PROCEDURE.** Schema change taken mid-pilot on 2026-08-14 (§2.3).
  Needs sign-off; it is a decision taken after seeing first results and must be
  declared as such.
- **PROCEDURE recall.** Student recall is 28–35 % on this label, which carries
  the highest-effect-size risk factors (CVC aOR 35.3, urinary catheter
  aOR 37.1). Scope a fix or declare it a limitation.
- **IAA subset.** 5–8 notes, second annotator, independent, no pre-annotation.
  Target: pairwise entity-level F1 ≥ 0.75 exact / ≥ 0.85 partial, per-type not
  pooled; κ retained only as a secondary comparability figure. Second annotator
  still to be identified. Fallback if none is found: re-annotate a subset after
  a delay and report **intra**-annotator agreement, explicitly labelled as such.