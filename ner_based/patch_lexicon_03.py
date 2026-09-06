#!/usr/bin/env python
"""Lexicon patch 3 (2026-08-26): new feature 48 + acronym misses.

A. NEW FEATURE other_indwelling_device.
   The corpus contains ~2,700 mentions of indwelling hardware that the
   literature-derived feature list does not cover: lvad 740, peg 690,
   rvad 476, chest tube 361 + chest tube placement 500, ngt 352, tpn 330.
   Added as a single catch-all rather than as separate features, because the
   risk-factor review supports the mechanism (indwelling foreign body as a
   portal of entry) but not MRSA-specific effect sizes for each device.
   Provenance differs from features 1-47 and must be stated as such.

B. `oht` 364 is orthotopic HEART TRANSPLANT, not a device -> goes to
   immunosuppressed_state with the other transplant acronyms.

C. Acronym misses found in the rows 120-300 sweep.

Run from ner_based/ with lexicon.yaml present.
"""
import pathlib, sys

p = pathlib.Path("lexicon.yaml"); src = p.read_text(); edits = 0

def sub(old, new, label):
    global src, edits
    if src.count(old) != 1:
        sys.exit(f"! {label}: anchor matched {src.count(old)} times, aborting")
    src = src.replace(old, new); edits += 1; print(f"  ok  {label}")

# ── A. new feature, appended after supplemental_oxygen ───────────────────
sub("""      `nasal canula` is a corpus misspelling. Bare `\\bnc\\b` is retained but
      `no complaints` is excluded.""",
    """      `nasal canula` is a corpus misspelling. Bare `\\bnc\\b` is retained but
      `no complaints` is excluded.

  - name: other_indwelling_device
    label: Other indwelling device (corpus-derived, not literature-derived)
    entity_types: [PROCEDURE]
    temporality: ignored
    include:
      - '\\blvad\\b|\\brvad\\b|\\bbivad\\b|ventricular assist'
      - '\\bpeg\\b|gastrostomy|\\bgtube\\b|\\bg[- ]tube\\b|jejunostomy|\\bj[- ]tube\\b'
      - '\\bngt\\b|nasogastric|\\bogt\\b|orogastric'
      - 'chest tube|thoracostomy|\\bpleurx\\b'
      - '\\bjp drain\\b|\\bdrain\\b|penrose'
      - '\\btpn\\b|parenteral nutrition'
      - 'colostomy|ileostomy|urostomy|\\bostomy\\b'
      - 'ventriculoperitoneal|\\bvp shunt\\b|\\bevd\\b'
      - 'ureteral stent|nephrostomy|biliary stent|\\bptbd\\b'
    exclude:
      - 'drainage of'
      - 'no drain'
    notes: >
      PROVENANCE DIFFERS FROM FEATURES 1-47. Those come from the risk-factor
      literature review; this one is corpus-derived. The corpus contains ~2,700
      mentions of indwelling hardware with no home in the reviewed feature set
      (lvad 740, peg 690, rvad 476, chest tube 861, ngt 352, tpn 330). The
      mechanism is supported -- an indwelling foreign body is a portal of entry,
      and TPN requires central access -- but MRSA-specific effect sizes are not
      established per device, which is why they are pooled into one column
      rather than split. State this distinction in the methods.

      entity_types is PROCEDURE-only, which is load-bearing: `peg` would
      otherwise match `polyethylene glycol` (1486 occurrences, MEDICATION).
      Central lines, urinary catheters, dialysis access, tracheostomy and
      ventilation have their own features and are not duplicated here.""",
    "NEW feature: other_indwelling_device")

# ── B. oht is a heart transplant ─────────────────────────────────────────
sub("""      - '\\bolt\\b|\\bddkt\\b|\\blrkt\\b|\\bsot\\b'""",
    """      - '\\bolt\\b|\\bddkt\\b|\\blrkt\\b|\\bsot\\b|\\boht\\b|\\bolt\\b'""",
    "immunosuppressed_state: add oht (heart transplant)")

# ── C. acronym misses ────────────────────────────────────────────────────
sub("""      - 'methotrexate'
      - 'basiliximab|anti[- ]thymocyte|thymoglobulin'""",
    """      - 'methotrexate|\\bmtx\\b'
      - '\\bfk\\b|\\bfk-?506\\b'
      - 'basiliximab|anti[- ]thymocyte|thymoglobulin'""",
    "immunosuppressant: mtx, fk")

sub("""      - '\\besrd\\b'
      - '\\bckd\\b'""",
    """      - '\\besrd\\b'
      - '\\bckd\\b|\\bckd[1-5]\\b'""",
    "renal_failure: ckd3 closed form")

sub("""      - '\\bdm2?\\b|\\bt1dm\\b|\\bt2dm\\b|\\biddm\\b|\\bniddm\\b'""",
    """      - '\\bdm[12]?\\b|\\bt1dm\\b|\\bt2dm\\b|\\biddm\\b|\\bniddm\\b'
      - '\\bdka\\b|\\bhhs\\b'""",
    "diabetes: dm1, dka")

sub("""      - '\\bcabg\\b|\\bpci\\b|\\bavr\\b|\\bmvr\\b|\\btavr\\b'""",
    """      - '\\bcabg\\b|\\bpci\\b|\\bavr\\b|\\bmvr\\b|\\btavr\\b'
      - '\\bbka\\b|\\baka\\b'""",
    "surgery: bka/aka")

sub("""      - '\\bdlbcl\\b|\\bnhl\\b|\\bhl\\b|\\bmds\\b|\\bmgus\\b'""",
    """      - '\\bdlbcl\\b|\\bnhl\\b|\\bhl\\b|\\bmds\\b|\\bmgus\\b'
      - '\\batll\\b|\\bdcis\\b|\\blcis\\b|\\brcc\\b|\\bhcc\\b'
      - '\\bca\\b(?= *\\(|$)|prostate ca|breast ca|lung ca'""",
    "malignancy: atll, dcis, prostate ca")

p.write_text(src); print(f"\n{edits} edits applied")
