MRSA NLP — Annotation Guidelines (v2)

Author: Tobias Rademacher Status: active — use this version for the gold set

1. Purpose

These guidelines define how to annotate clinical notes from AIR·MS for training and evaluating a named entity recognition (NER) model. The model's goal is to extract clinical signals relevant to MRSA bacteremia risk prediction.

Two distinct uses:

Silver corpus — llama3.1:70b pre-annotates notes using a compressed form of these rules (src/ner/preannotate.py); a human corrects them in INCEpTION.
Gold set — 25 held-out notes annotated by hand with no pre-annotation, used to measure how good the silver corpus and the student model actually are.

Governing principle. Annotation records span + label only. Whether a mention is negated, historical, hypothetical, or about someone other than the patient is recorded nowhere in the annotation — it is resolved later by a rule-based medspaCy ConText layer at inference time. Tag the span regardless of context; the context question belongs to a different component.

2. Entity types

Three labels. Anything that does not fit one of the three is left unlabeled.

2.1 DISEASE

Any clinical condition, diagnosis, symptom, syndrome, or infection.

Includes

Named diseases: pneumonia, rheumatoid arthritis, MRSA bacteremia
Symptoms: fever, hypotension, shortness of breath
Conditions: immunosuppression, renal failure, diabetes mellitus
Wounds and skin findings: decubitus ulcer, cellulitis, chronic wound
Standard abbreviations: RA, CHF, DM2, UTI, COPD, ICH
Prior infections: MRSA bacteremia in "history of MRSA bacteremia (2021)"

Excludes

Body parts alone: left arm, L leg, LLE
General descriptors: ill, sick, unwell, not doing well
Lab and test names or values — see §2.4
2.2 MEDICATION

Any drug, drug class, or pharmacological agent named in the note.

Includes

Drug names: vancomycin, prednisone, methotrexate, tacrolimus
Drug classes: corticosteroids, antibiotics, immunosuppressants
Abbreviations and short forms: abx, vanc, pip-tazo
Allergens that are drugs (§3.4)

Excludes

Dosage, strength, form, route, frequency — §4
Vitamins and supplements unless clearly therapeutic
2.3 PROCEDURE

Any medical procedure, intervention, or indwelling device. Devices and lines are procedures in this schema — this is the most-missed category, so read the list.

Includes

Devices and lines: central line, PICC, tunneled cath, CVC, Foley, urinary catheter, tracheostomy
Procedures: intubation, hemodialysis / HD, CABG, wound debridement, surgery, bone marrow transplant, lumbar puncture
Abbreviations: BAV, RHC, CVVH, OHT, BMT

Excludes

Routine assessments: physical exam, vitals
Non-invasive imaging: CT scan, chest X-ray, TTE (invasive procedures such as lumbar puncture and cardiac catheterization are included)
2.4 Laboratory tests and results — NOT annotated

Lab names, panels and values are not entities in this schema. They are available as structured EHR data and add nothing here.

WBC 12.3, blood cultures, CBC, creatinine 2.1, INR → unlabeled
But a diagnosis stated from a lab is DISEASE: bacteremia, neutropenia
3. Edge cases
3.1 Negation — annotate

Tag the entity even when explicitly negated. Do not tag the negation cue.

"No pneumonia on imaging" → annotate pneumonia
"Fever denied" → annotate fever
"Ruled out for endocarditis" → annotate endocarditis

Rationale: negation is detected downstream by ConText. Withholding the span would destroy the information the assertion layer needs.

3.2 Abbreviations — annotate as the type they represent

MRSA → DISEASE · PICC → PROCEDURE · RA → DISEASE · vanco → MEDICATION

Where both an abbreviation and its expansion appear, tag them as two separate spans — do not merge across the parenthesis (§4).

"ICH (intracerebral hemorrhage)" → two DISEASE spans

If unsure what an abbreviation means, leave it unlabeled and record it in §5.

3.3 Speculation / uncertainty — annotate

Tag the entity; do not tag the hedge.

"Rule out sepsis" → annotate sepsis
"Possible pneumonia" → annotate pneumonia
"Concern for line infection" → annotate line infection
3.4 Allergies — annotate the drug
"Penicillin allergy" → annotate Penicillin as MEDICATION
"Allergic to sulfa" → annotate sulfa as MEDICATION

Do not annotate the word "allergy" itself. Whether the drug was administered or merely listed as an allergy is an assertion-layer distinction, not an entity one.

3.5 Historical mentions — annotate normally
"History of MRSA bacteremia in 2021" → annotate MRSA bacteremia
"s/p BAV (5/27/14)" → annotate BAV as PROCEDURE

Do not include the date in the span.

3.6 Family history — annotate
"Mother had breast cancer" → annotate breast cancer as DISEASE
"Father with MRSA wound infection" → annotate MRSA wound infection

Experiencer (patient vs. relative) is resolved by ConText downstream.

3.7 Compound entities — one span per clinical concept

Tag the complete clinical term as a single span; do not split a term that names one concept.

MRSA bacteremia → one DISEASE span, not MRSA + bacteremia
central venous catheter → one PROCEDURE span
acute kidney injury → one DISEASE span

This is bounded by §4: the span is the clinical term itself, with decoration, dosing and trailing anatomy excluded.

3.8 Coordinated entities — annotate separately
"Fever and chills" → two DISEASE spans
"Vanc + pip-tazo" → two MEDICATION spans
3.9 Repeated mentions — annotate every occurrence

If LVAD appears 44 times in a note, all 44 are annotated. Entity counts in this project therefore report occurrences, not distinct entities.

4. Span boundary rules

Principle: tag the shortest span that carries the full clinical meaning.

Always exclude

Articles and determiners: "a central line" → central line
Severity modifiers: "severe pneumonia" → pneumonia
Trailing punctuation
Leading bullets, numbering, and section labels: "• Hyperlipidemia" → Hyperlipidemia

MEDICATION — drug name only. Exclude parenthetical brand names, dose, strength, form, route and frequency.

"omeprazole (PRILOSEC) 20 mg capsule Take 20 mg by mouth daily" → omeprazole
"fluticasone-salmeterol (ADVAIR) 250-50 mcg" → fluticasone-salmeterol

DISEASE / PROCEDURE — full clinical term, minus trailing anatomical qualifiers that are separate body-part mentions.

"Macular degeneration of right eye" → Macular degeneration
"Lumpectomy Right" → Lumpectomy
"left thalamic ICH (2010)" → ICH

Abbreviation and expansion are separate spans.

"AS (aortic stenosis)" → AS and aortic stenosis, not one merged span
5. Questions log

Record every judgment call made while annotating. This log is the source of future guideline versions and is part of the assessed deliverable.

Note ID	Excerpt	What I labeled	Why uncertain	Revisit?
				
				
6. Out of scope (deliberate, for the limitations section)

Not annotated in v2; recorded here so their absence is a documented choice rather than an oversight.

Assertion / temporality / experiencer — negated, historical, hypothetical, family-member status. Handled downstream by rule-based medspaCy ConText (cf. Harkema et al. 2009). Currently unvalidated, because no gold attributes exist.
SEVERITY as a fourth label (immunocompromised, critically ill) — optional in the exposé, dropped to keep the label set at three.
Entity linking to SNOMED CT / RxNorm (exposé Phase 4).
Healthcare-context factors — prior hospitalization, nursing-home exposure, facility transfer. Reliably coded in structured EHR fields; excluded from NER per the risk-factor master list (rows 22–25).
7. Open questions for supervision
Family history (§3.6). A family-history MRSA mention may be genuinely risk-relevant rather than noise. Confirm the assertion-layer treatment.
Allergies (§3.4). Confirm that an allergy mention should produce a MEDICATION span that ConText later marks as not-administered.
PROCEDURE recall. Student recall is 28–35 % on this label, which carries the highest-effect-size risk factors (CVC aOR 35.3, urinary catheter aOR 37.1). Scope a fix or declare it a limitation.
IAA subset. 5–8 notes, second annotator, independent, no pre-annotation. Target κ ≥ 0.75. Second annotator still to be identified.