# MRSA NLP — Annotation Guidelines (v1, DRAFT)

**Author:** Tobias Rademacher

## 1. Purpose

These guidelines define how to annotate clinical notes from AIR.MS for training a
named entity recognition (NER) model. The model's goal is to extract clinical
signals relevant to MRSA bacteremia risk prediction.

## 2. Entity types

We label THREE entity types. Anything that does not fit one of these three is
left unlabeled.

### DISEASE
Any clinical condition, diagnosis, symptom, or syndrome.

Includes:
- Named diseases: "pneumonia", "rheumatoid arthritis", "MRSA bacteremia"
- Symptoms: "fever", "hypotension", "shortness of breath"
- Conditions: "immunosuppression", "renal failure", "diabetes"
- Standard abbreviations: "RA" (rheumatoid arthritis), "CHF", "DM2", "UTI"
- Prior infections relevant to risk: "history of MRSA", "prior C. diff"

Excludes:
- Body parts alone ("left arm" — not a disease)
- Lab values or vital signs ("WBC 12.3" — not a disease)
- General descriptors ("ill", "sick", "unwell")

### MEDICATION
Any pharmacological agent administered to or prescribed for the patient.

Includes:
- Drug names: "vancomycin", "prednisone", "methotrexate", "tacrolimus"
- Drug classes: "corticosteroids", "antibiotics", "immunosuppressants"
- Abbreviations: "abx" (antibiotics)

Excludes:
- Dosages and frequencies ("10mg", "twice daily") — annotate only the drug name
- Allergies ("penicillin allergy") — see edge cases below
- Vitamins and supplements unless explicitly therapeutic

### PROCEDURE
Any medical procedure, intervention, or indwelling device.

Includes:
- Devices: "central line", "PICC line", "Foley catheter", "urinary catheter"
- Procedures: "intubation", "hemodialysis", "surgery", "bone marrow transplant"
- Abbreviations: "CVC" (central venous catheter), "BMT"

Excludes:
- Routine assessments ("physical exam", "vitals")
- Imaging unless clinically invasive ("CT scan" — no; "lumbar puncture" — yes)

## 3. Edge cases (decisions for v1)

### 3.1 Negation

### 3.2 Abbreviations
**Decision: annotate abbreviations as the entity type they represent.**

"MRSA" → DISEASE (or part of "MRSA bacteremia" as DISEASE)
"PICC" → PROCEDURE
"RA" → DISEASE
"vanco" → MEDICATION

If unsure what an abbreviation means, leave it unlabeled and add to questions log.

### 3.3 Speculation / uncertainty
**Decision: annotate the entity; do not annotate the speculation word.**

"Rule out sepsis" → annotate "sepsis" as DISEASE.
"Possible pneumonia" → annotate "pneumonia" as DISEASE.

### 3.4 Allergies
**Decision: do NOT annotate the drug name in an allergy context.**

"Penicillin allergy" → do not annotate "penicillin".
"Allergic to sulfa" → do not annotate "sulfa".

Rationale: the drug isn't being given, so it's not a meaningful medication feature.

### 3.5 Historical mentions
**Decision: annotate normally.**

"History of MRSA bacteremia in 2021" → annotate "MRSA bacteremia" as DISEASE.

Prior infections are clinically relevant for MRSA risk.

### 3.6 Family history
**Decision: do NOT annotate.**

"Mother had breast cancer" → do not annotate "breast cancer".

We only want patient-specific information.

### 3.7 Compound entities
**Decision: annotate the longest meaningful clinical span.**

"MRSA bacteremia" → annotate the full "MRSA bacteremia" as one DISEASE entity,
NOT "MRSA" and "bacteremia" separately.

"Central venous catheter" → annotate as one PROCEDURE.

### 3.8 Coordinated entities
**Decision: annotate each entity separately.**

"Fever and chills" → annotate "fever" and "chills" as two separate DISEASE entities.

## 4. Span boundary rules

- Include the full clinical term, exclude surrounding articles and modifiers
  - "a central line" → annotate "central line", not "a central line"
  - "severe pneumonia" → annotate "pneumonia", not "severe pneumonia"
    (severity is captured elsewhere; the entity is the condition itself)
- Do not include trailing punctuation
- Do not include dosage or frequency in MEDICATION spans

## 5. Questions log

| Note ID | Excerpt | What I labeled | Why uncertain | Revisit? |
|---|---|---|---|---|
|---|---|---|---|---|
|---|---|---|---|---|



## 6. Fututre work?

These may be added later:
- SEVERITY (immunocompromised, critically ill) — optional from exposé
- Entity linking to SNOMED CT / RxNorm (Phase 4)
- Temporal qualifiers (acute vs. chronic)