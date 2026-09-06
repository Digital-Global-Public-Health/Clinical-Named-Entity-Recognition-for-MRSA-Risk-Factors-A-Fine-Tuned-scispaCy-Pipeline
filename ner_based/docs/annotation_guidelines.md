# MRSA NLP — Annotation Guidelines

**Author:** Tobias Rademacher

These guidelines define how clinical notes from AIR·MS are annotated for named
entity recognition in support of MRSA bacteremia risk extraction.

Two uses. The **silver corpus** is annotated by `llama3.3:70b` from a compressed
form of these rules (`src/ner/preannotate.py`). The **gold set** is annotated by
hand with no machine assistance and measures the silver corpus and the student
model: four test-split patients × four note types = 16 notes.

**Governing principle.** Entity annotation records span and label only. Whether
a mention is negated, historical, hypothetical or about someone other than the
patient is not part of the entity decision: tag the span regardless of context,
and never tag the cue itself. For the gold set only, a second pass records those
judgments as attributes on the already-annotated spans (§3).

---

## 1. Entity types

Three labels. Anything that does not fit one of them is left unlabeled.

### 1.1 DISEASE

Any clinical condition, diagnosis, symptom, syndrome or infection.

**Include:** named diseases (pneumonia, MRSA bacteremia); symptoms (fever,
hypotension); conditions (renal failure, immunosuppression); wounds and skin
findings (decubitus ulcer, cellulitis); standard abbreviations (RA, CHF, DM2,
UTI, COPD); prior infections; substance-use disorder terms and noun-phrase
substance mentions (`crack use`, `substance use disorder`).

**Exclude:** body parts alone (left arm, LLE); general descriptors (ill, unwell);
laboratory names and values (§1.4); consumption described as an activity with a
quantity (`drinking 2-3 beers daily`).

### 1.2 MEDICATION

Any drug, drug class or pharmacological agent named in the note.

**Include:** drug names; drug classes, with or without a specific agent
(`corticosteroids`, `abx`, `antimicrobial`); abbreviations and short forms
(`vanc`, `pip-tazo`); drug allergens (§2.4); contrast agents and administered
diagnostic substances (`Gadobutrol`, `perflutren`, `sulfur hexafluoride`,
`iodine`), which are pharmacological agents given to the patient and are
filtered out of the feature layer downstream rather than at annotation.

**Exclude:** dosage, strength, form, route and frequency (§4); vitamins and
supplements unless clearly therapeutic; non-pharmacological allergens
(`shellfish`, `latex`, `dust`), which are not pharmacological agents and are
left unlabeled, so no allergy attribute applies; diluents — `sodium chloride
0.9 %` as an IV vehicle is skipped, `sodium chloride 1 gram tablet` is tagged.

A class must name a *pharmacological* class. `IV fluids` and `IVF` name a route
and a physical form and are not tagged.

### 1.3 PROCEDURE

Any medical procedure, intervention or indwelling device. Devices and lines are
procedures in this schema; this is the most-missed category.

**Include:** devices and lines (central line, PICC, CVC, Foley, urinary
catheter, tracheostomy); procedures (intubation, hemodialysis, CABG, wound
debridement, surgery, lumbar puncture); abbreviations (BAV, RHC, CVVH, BMT);
non-invasive imaging (CT, chest X-ray, TTE, MRI, CTA); immunisation.

Non-invasive imaging is included because the invasive/non-invasive boundary
carries almost no surface signal for a tok2vec model — `CTA` and `CVC` sit in
near-identical contexts — and feature aggregation filters to specific
risk-factor concepts downstream, so additional imaging entities cost nothing
there.

**Exclude:** routine assessments (physical exam, vitals); care-coordination
activity (`SNF placement`, `SW consult`, `pain management consult`). Long-term
care exposure is a risk factor, but one recorded reliably in structured fields
and deliberately outside this schema (§5).

### 1.4 Laboratory tests and results — not annotated

Lab names, panels and values are available as structured data and add nothing
here: `WBC 12.3`, `blood cultures`, `CBC`, `creatinine 2.1` are unlabeled. A
diagnosis stated from a lab is DISEASE: `bacteremia`, `neutropenia`.

---

## 2. Entity decisions

**2.1 Negation — tag the entity, never the cue.** `No pneumonia on imaging` →
tag `pneumonia`. Negation is recorded on the polarity axis (§3.1); withholding
the span would destroy what the assertion layer needs.

Negated but named → tag (`no LAD`, `negative for valvular vegetation`). Normal
descriptors naming nothing → skip (`VF intact`, `unremarkable`, `patent`).
Negative-polarity functional deficits → skip (`inability to abduct`,
`non-tender`); positively-named equivalents are tagged (`vision loss`, `CN VI
palsy`). Generic negatives → skip (`no acute abnormality`); specific negated
findings → tag (`no pneumothorax`). Slash-compressed multi-condition
abbreviations → skip (`no m/r/g`); tag when spelled out.

**2.2 Abbreviations — tag as the type they represent.** `MRSA` → DISEASE,
`PICC` → PROCEDURE, `vanco` → MEDICATION. Where both an abbreviation and its
expansion appear, tag two separate spans: `ICH (intracerebral hemorrhage)`.
Where an abbreviation cannot be resolved with confidence from the note or from
standard usage, leave it unannotated rather than guess, and record it in §6.

**2.3 Speculation — tag the entity, not the hedge.** `Rule out sepsis` → tag
`sepsis`. Recorded on the certainty axis (§3.2).

**2.4 Allergies — tag the drug.** `Penicillin allergy` → tag `Penicillin`. Do
not tag the word "allergy". Whether the drug was administered or listed as an
allergen is recorded on the allergy axis (§3.5). Non-pharmacological allergens
are not annotated at all (§1.2). An allergy field naming no agent
(`Contrast Allergy?: No`) is not annotated, because there is no MEDICATION span
for the attribute to attach to.

**Prophylaxis: skip the header, tag the drug.** `DVT PPx` names a condition
being prevented; tagging `DVT` would assert a diagnosis the patient does not
have. Tag `Lovenox` instead. By the same logic, immunisation is tagged as a
PROCEDURE but the disease it targets is not: `fully vaccinated for COVID` → tag
the vaccination only.

Order titles are the opposite case and the condition **is** tagged:
`Dysphagia Puree diet` → tag `Dysphagia`, because the diet treats a condition
the patient has. Both occurrences of a repeated order title are tagged.

**2.5 Historical mentions — tag normally**, without the date.
`History of MRSA bacteremia in 2021` → tag `MRSA bacteremia`.

**2.6 Family history — tag normally.** `Mother had breast cancer` → tag
`breast cancer`. Experiencer is recorded on its own axis (§3.4).

**2.7 Compound entities — one span per clinical concept.** `MRSA bacteremia`,
`central venous catheter` and `acute kidney injury` are each a single span,
bounded by §4.

**2.8 Coordinated entities — separate spans.** `Fever and chills` → two spans.

**2.9 Repeated mentions — tag every occurrence.** Entity counts in this project
therefore report occurrences, not distinct entities.

**2.10 Structured symptom scales — tag every symptom name**, including items
scored zero. The zero is an assertion of absence and is recorded on the polarity
axis.

**2.11 Misspellings — tag verbatim**, never corrected. Hydromorphone appears in
five surface forms across the pilot notes (`dilaudid`, `Dilaudid`, `Dailudid`,
`Diladudid`, `DILAUDID`); also `zoysn` alongside the correct `zosyn`, `opiods`,
`subtherapuetic`, `tachypnic`, `mrsa bactermia`. This is a countable source of
NER difficulty that additional training data does not remove.

**2.12 Pressure-injury ellipsis — tag, with stage and site.**
`+stage II sacrum/coccyx`, `+stage II right heel` → DISEASE. §1.1 lists
decubitus ulcer explicitly, and the standalone-interpretability test of §4
applies to the span taken rather than to a fragment of it. A stage without a
site is not tagged.

---

## 3. Assertion attributes (gold set only)

Recorded as span-layer features in INCEpTION, in one pass over the spans already
annotated under §1–§2 and §4. Five independent axes, each with a default; most
spans are all-default and only exceptions are changed. A span may be negated and
historical (`no prior MRSA`), or allergic and negated (`denies penicillin
allergy`).

| axis | values | default |
|---|---|---|
| `polarity` | affirmed · negated | affirmed |
| `certainty` | certain · uncertain · hypothetical | certain |
| `temporality` | current · historical | current |
| `experiencer` | patient · other | patient |
| `allergy` | no · yes | no |

### 3.1 polarity

`negated` when the note states the finding is not present: `no pneumonia on
imaging`, `fever denied`, `negative for valvular vegetation`, `ruled out for
endocarditis` (past tense, resolved). A reassuring finding that names something
present is affirmed: `wound clean and dry`.

Negation attached to *change* does not negate the entity: `no significant
interval change in the fluid collection` → the collection is affirmed. A failed
procedure is affirmed, because the attempt occurred. `did not help` negates
efficacy, not administration: `oxycodone did not help` → oxycodone affirmed.
ESAS items scored `0-none` are negated.

### 3.2 certainty

`uncertain` — present but unconfirmed at the time of writing: `possible
pneumonia`, `concern for sepsis`, `may represent abscess`. Both members of a
differential are uncertain (`pneumonia vs atelectasis`), unless one is
independently affirmed elsewhere in the note. A hedge on a causal explanation
propagates to the entities inside it. Negative-polarity hedges are uncertain
rather than negated: `unlikely to be infective endocarditis` stops short of
assertion. `rule out` and `r/o` are uncertain; past-tense `ruled out` is
negation. Text that declines to assert presence is uncertain whether the wording
is clinician prose or an ICD code description: `macular edema presence
unspecified`.

`hypothetical` — conditional, planned or future-framed, and not claimed to exist:
`if she develops fever, start vancomycin`; `planned for surgery on 1/12`;
`may consider Haldol`. Monitoring lists are hypothetical: `monitor for sedation,
confusion, hallucinations`.

PRN indications are **affirmed**, not hypothetical: `as needed for cough`
describes when to take a drug, not whether the condition exists. For some
patients a medication-list indication is the only place a symptom is documented.

### 3.3 temporality

**Note-relative.** The note extract carries no outcome date, so `historical`
means the note asserts the mention predates the current episode, judged from the
text alone. This is weaker than the temporal separation a prior-MRSA feature
ultimately requires, and is a documented limitation.

`historical` when the text marks it as past: `h/o MRSA bacteremia`, `s/p BAV
(5/27/14)`, `prior line infection`, `resolved`, `treated in 2019`. Absence of a
history cue means current; do not infer historicity from clinical plausibility.

A history cue scopes over a whole coordinated list: `PMHx of ADHD, bipolar
disorder, crack cocaine use` → all three historical. `s/p` inside the current
encounter stays current; `s/p` with a past date is historical. A diagnosis with
a past date but active treatment is current, since the date attaches to the
diagnosis event rather than to the condition's persistence. Presenting
complaints marked `h/o … x 1 day` are current. Comparison studies in radiology
reports are historical even when dated a day earlier.

Narrative tense and bare dates are history cues for the annotator but not for
ConText: `he returned 12/26`, `was put on`. Annotate historical and expect the
pipeline to miss them.

### 3.4 experiencer

`other` when the mention belongs to a relative or another person: `mother had
breast cancer`, `FHx significant for diabetes`. The bare word *Family* in a
provenance line (`Source: Patient, Family and Team`) does not make the
surrounding findings family-experienced; those spans stay `patient`.

### 3.5 allergy

MEDICATION spans only. `yes` when the drug is named as something the patient
reacts to rather than something administered, by either of two shapes: an inline
cue (`allergic to vancomycin`, `PCN allergy`) or an allergy list or header
region, where the drug appears under an `Allergies:` header with no local cue.

Not `yes`: `allergic rhinitis`, `allergy testing`, `seasonal allergies` — these
are diseases or procedures. `denies penicillin allergy` is `allergy=yes` **and**
`polarity=negated`; neither reading supports the drug having been administered.

Allergy status is a property of the patient rather than the note, and the four
patients in the gold grid had few recorded drug allergies between them. The
allergy component is therefore evaluated on a corpus sample rather than on gold.

### 3.6 Not recorded

Severity and acuity (`severe`, `acute`) are stripped from spans per §4 and are
not an axis. Section membership is descriptive metadata only and nothing
downstream reads it.

---

## 4. Span boundary rules

**Principle: tag the shortest span that carries the full clinical meaning.**

**Always exclude** articles and determiners; severity and acuity modifiers
(`severe pneumonia` → `pneumonia`); trailing punctuation; leading bullets,
numbering and section labels.

**Always keep** polarity modifiers that constitute the finding: `abnormal
intracranial enhancement`, `increased work of breathing`.

**MEDICATION — drug name only.** Exclude parenthetical brand names, dose,
strength, form, route and frequency: `omeprazole (PRILOSEC) 20 mg capsule Take
20 mg by mouth daily` → `omeprazole`.

**DISEASE and PROCEDURE — the full clinical term**, minus trailing anatomical
qualifiers that are separate body-part mentions: `Macular degeneration of right
eye` → `Macular degeneration`.

**The anatomy connector test.** Adjectival anatomy inside the term stays
(`abdominal pain`, `orbital cellulitis`); anatomy introduced by `of`, `in`, `at`
or `involving` is dropped (`pain in his R eye` → `pain`). Laterality is always
dropped.

**The standalone-interpretability carve-out.** The connector test strips anatomy
only when the remaining head noun is a self-sufficient clinical finding. Bare
morphological descriptors — `curvature`, `stenosis`, `atrophy`, `thickening`,
`calcification` — name a shape rather than a condition, and the anatomy is what
makes them clinical: `stenosis of the left renal artery` is kept whole. A model
trained on `curvature` as a DISEASE is learning noise.

**Abbreviation and expansion are separate spans:** `AS (aortic stenosis)` → two.

---

## 5. Out of scope

Recorded so that their absence is a documented choice.

- **Assertion, temporality and experiencer** are not part of *entity*
  annotation. They are handled downstream by rule-based medspaCy ConText and an
  AIR·MS allergy component, and measured against the gold attributes of §3.
- **Section membership.** The sectionizer assigns a category as descriptive
  metadata. Its boundaries leak, because the export contains no newlines and a
  section therefore runs from its header to the next *matched* header.
- **SEVERITY as a fourth label** (immunocompromised, critically ill), to keep
  the label set at three.
- **A SOCIAL label**, which would resolve the substance-use boundary in §1.1
  more cleanly than folding disorder terms into DISEASE. A fourth label is a
  schema change, and PROCEDURE is already the thin one.
- **Entity linking** to SNOMED CT or RxNorm.
- **Healthcare-context factors** — prior hospitalisation, nursing-home exposure,
  facility transfer. These are recorded reliably in structured EHR fields, and
  extracting them from narrative would yield a less reliable copy of data
  already available.

---

## 6. Known limitations of the schema

- **Held medications.** `[Held by provider] heparin` is not administered, but no
  axis records this, so it reads as administered in the feature layer.
- **`c/w` is ambiguous.** Before a treatment it reads "continue with"; before a
  finding, "consistent with". Both are affirmed, but a second annotator may
  split them.
- **Exam shorthand.** `No CVAT` and `No D/C Noted` are specific negated findings
  and are tagged, but sit adjacent to compressed normal descriptors `N/T`,
  `N/M`, `N/E`, which are skipped. The boundary is thin.
- **Inter-annotator agreement.** No second annotator was available, so agreement
  is not reported and every accuracy figure rests on one annotator's judgment.
- **Unresolved abbreviations.** `MSMS` appears roughly four times and
  consistently occupies the position of a facility or service name; it is not
  tagged, and on that reading falls outside the schema in any case.