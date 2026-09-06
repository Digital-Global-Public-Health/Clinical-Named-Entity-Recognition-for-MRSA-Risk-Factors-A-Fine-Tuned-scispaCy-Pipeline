#!/usr/bin/env python
"""Apply the five lexicon fixes found in the first corpus review (2026-08-26).

1. myocardial_infarction: drop the CAD patterns. `cad` 4410 + variants was
   ~4900 of 8450 occurrences, but coronary artery disease is a chronic
   condition, not an infarction. The spreadsheet parallels CCI_MI_90d, which
   counts MI specifically.
2. surgery: drop the bare `otomy` suffix (caught `phlebotomy` 38, a blood
   draw); require a word boundary before `operative` so `postoperative`
   no longer matches; exclude donor procedures and postoperative
   complications (`seroma, postoperative` 189, `postoperative pain` 51 are
   DISEASE spans, not operations); drop the over-broad bare `\\bor\\b`.
3. malignancy: add `dlbcl` (1541 occurrences, unclaimed -- the `lymphoma`
   pattern misses the acronym) and other common haem-onc acronyms.
4. immunosuppressed_state: add `gvhd` (1086, unclaimed) -- graft-versus-host
   disease implies allogeneic transplant and ongoing immunosuppression.
5. malignancy: drop bare `\\bALL\\b`, which matches the English word.

Run from ner_based/ with lexicon.yaml present.
"""

import pathlib
import sys

p = pathlib.Path("lexicon.yaml")
src = p.read_text()
edits = 0


def sub(old, new, label):
    global src, edits
    if src.count(old) != 1:
        sys.exit(f"! {label}: anchor matched {src.count(old)} times, aborting")
    src = src.replace(old, new)
    edits += 1
    print(f"  ok  {label}")


# ── 1. myocardial_infarction: MI only, not CAD ────────────────────────────
sub("""      - 'myocardial infarction'
      - '\\bmi\\b'
      - '\\bstemi\\b|\\bnstemi\\b'
      - 'heart attack'
      - '\\bcad\\b'
      - 'coronary artery disease'
    exclude: []""",
    """      - 'myocardial infarction'
      - '\\bmi\\b'
      - '\\bstemi\\b|\\bnstemi\\b'
      - 'heart attack'
    exclude:
      - 'mitral'
      - '\\bmri\\b'""",
    "myocardial_infarction: drop CAD")

sub("""      Bare `\\bmi\\b` is risky -- check corpus review output. `Heart Attack`
      appears in a family-history table in gold, so the experiencer axis
      matters for this feature.""",
    """      CAD patterns removed after review: `cad` 4410 + `coronary artery
      disease` 873 + variants were ~4900 of 8450 occurrences, but CAD is a
      chronic condition, not an infarction, and the parallel structured
      feature (CCI_MI_90d) counts MI specifically. If a CAD feature is wanted
      it should be its own column. Remaining: mi 1496, myocardial infarction
      268, nstemi 209, heart attack 85. `Heart Attack` appears in a
      family-history table in gold, so the experiencer axis matters here.""",
    "myocardial_infarction: note")

# ── 2. surgery: suffix and postoperative fixes ────────────────────────────
sub("""      - 'surgery|surgical|operative|\\bor\\b procedure'
      - 'resection|excision|amputation|arthroplasty'
      - 'laparotomy|laparoscop|thoracotomy|craniotomy|sternotomy'
      - '\\bs/p\\b.*(ectomy|otomy|plasty|ostomy)'
      - '(ectomy|otomy|plasty)\\b'
      - 'surgical site infection|\\bssi\\b'
      - 'post[- ]operative wound'
    exclude:
      - 'surgical history'
      - 'no surgery'""",
    """      - 'surgery|surgical'
      - '\\boperative\\b|\\bintraoperative\\b'
      - 'resection|excision|amputation|arthroplasty'
      - 'laparotomy|laparoscop|thoracotomy|craniotomy|craniectomy|sternotomy'
      - '(ectomy|plasty)\\b'
      - 'surgical site infection|\\bssi\\b'
      - 'post[- ]operative wound'
    exclude:
      - 'surgical history'
      - 'no surgery'
      - 'phlebotomy'
      - 'postoperative|post[- ]op\\b'
      - '\\bdonor\\b'
      - 'tracheostomy'""",
    "surgery: patterns")

sub("""      Two entity types because surgical site infection is a DISEASE while the
      operation is a PROCEDURE. The bare (ectomy|otomy|plasty) suffix pattern
      is broad -- review carefully. Bare `\\bor\\b` will over-match; consider
      dropping it.""",
    """      Two entity types because surgical site infection is a DISEASE while the
      operation is a PROCEDURE.

      Review fixes: the bare `otomy` suffix caught `phlebotomy` 38 (a blood
      draw) and is dropped; `operative` matched inside `postoperative`, so
      `seroma, postoperative` 189 and `postoperative pain` 51 -- both DISEASE
      complications, not operations -- are now excluded; `donor cardiectomy`
      73 is another patient's procedure; bare `\\bor\\b` removed as
      over-broad. `postoperative wound` is retained via its own pattern
      because it is a wound-care signal.

      OPEN DECISION: with temporality ignored, `hysterectomy` 190,
      `tonsillectomy` 41 and `hx toe amputation` 120 all count as surgery
      regardless of when they happened. The parallel structured feature is
      has_surgical_procedure_90d. Either set temporality: required here and
      accept 38.7 F1 on that axis, or state in the methods that this feature
      is ever-in-record and not window-comparable.""",
    "surgery: note")

# ── 3+5. malignancy: add haem-onc acronyms, drop bare ALL ─────────────────
sub("""      - '\\bca-p\\b|\\bAML\\b|\\bCLL\\b|\\bCML\\b|\\bALL\\b'
      - 'angiomyolipoma'
    exclude:
      - 'cancer screening'
      - 'family history of cancer'
    notes: 'Bare \\bALL\\b is dangerous; review before enabling.'""",
    """      - '\\bca-p\\b|\\baml\\b|\\bcll\\b|\\bcml\\b'
      - '\\bdlbcl\\b|\\bnhl\\b|\\bhl\\b|\\bmds\\b|\\bmgus\\b'
      - '\\ball\\b(?= *\\()|acute lymphoblastic'
      - 'angiomyolipoma'
      - 'blast(s|ic)? crisis|myelofibrosis'
    exclude:
      - 'cancer screening'
      - 'family history of cancer'
      - 'cancer risk'
    notes: >
      Bare \\bALL\\b removed -- it matches the English word. Acute
      lymphoblastic leukaemia is now caught by the spelled-out form or by
      `ALL (` where an expansion follows. Added after review: `dlbcl` 1541
      occurrences was completely unclaimed because the `lymphoma` pattern
      does not match the acronym.""",
    "malignancy: acronyms")

# ── 4. immunosuppressed_state: add GVHD ───────────────────────────────────
sub("""      - 'immunosuppress|immunocompromis|immunodeficien'
      - 'transplant'
      - 'neutropeni'
      - 'agranulocytosis'
      - 'asplen'
    exclude:
      - 'transplant evaluation'
    notes: 'Immunosuppressant DRUGS are a separate feature; both may fire.'""",
    """      - 'immunosuppress|immunocompromis|immunodeficien'
      - 'transplant'
      - 'neutropeni'
      - 'agranulocytosis'
      - 'asplen'
      - '\\bgvhd\\b|graft[- ]versus[- ]host'
      - '\\bscid\\b|common variable immunodef'
    exclude:
      - 'transplant evaluation'
      - 'transplant candidate'
    notes: >
      Immunosuppressant DRUGS are a separate feature; both may fire.
      Added after review: `gvhd` 1086 occurrences was unclaimed -- it is
      excluded from dialysis_access (correctly) but implies allogeneic
      transplant and ongoing immunosuppression, so it belongs here.
      NOT added: `itp` 1878 and `sarcoidosis` 1023. Both are usually
      steroid-treated, but that is immunosuppression by treatment rather than
      by definition -- the corticosteroid feature carries it.""",
    "immunosuppressed_state: gvhd")

p.write_text(src)
print(f"\n{edits} edits applied")
