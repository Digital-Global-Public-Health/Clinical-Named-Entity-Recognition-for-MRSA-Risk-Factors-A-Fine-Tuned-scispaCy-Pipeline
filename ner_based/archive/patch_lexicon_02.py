#!/usr/bin/env python
"""Lexicon fixes from the second corpus review (2026-08-26).

False positives:
  1. dialysis_access: `hd-mtx` 42 is high-dose methotrexate, not haemodialysis.
     Excluded here; methotrexate is already covered by immunosuppressant.
  2. central_venous_catheter: `midline catheter double lumen` 31 -- a midline
     terminates in the axilla and is peripheral by definition. Caught by the
     bare `double lumen` pattern; now excluded.
  3. supplemental_oxygen: CPAP and BiPAP deliver positive pressure, often on
     room air. They are not supplemental oxygen. Removed (1243 of the
     feature's 1525 occurrences), leaving nasal cannula / HFNC / NRB / venturi.

Missed surface forms found in the unmapped list:
  4. pneumonia: `pna` 798 -- the standard abbreviation, previously unclaimed.
  5. vancomycin_glycopeptide: `vanc` 595 -- the pattern was `vanco\\b`, which
     does not match the shorter clipping.
  6. immunosuppressed_state: `olt` 793 (orthotopic liver transplant) and
     `ddkt` 609 (deceased donor kidney transplant) -- both transplants, both
     missed because only the spelled-out word was matched.
  7. surgery: `cabg` 655 and `pci` 841 -- cardiac procedures with no
     ectomy/plasty suffix.

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


# ── 1. dialysis_access: exclude high-dose methotrexate ────────────────────
sub("""    exclude:
      - 'graft[- ]versus[- ]host'
      - 'coronary .*graft'
      - 'bypass graft'
      - 'dialysis flush'""",
    """    exclude:
      - 'graft[- ]versus[- ]host'
      - 'coronary .*graft'
      - 'bypass graft'
      - 'dialysis flush'
      - 'hd[- ]?mtx'
      - 'hd[- ]?methotrexate'""",
    "dialysis_access: exclude hd-mtx")

# ── 2. central_venous_catheter: exclude midlines ──────────────────────────
sub("""    exclude:
      - 'peripheral iv'
      - '\\bpiv\\b'""",
    """    exclude:
      - 'peripheral iv'
      - '\\bpiv\\b'
      - '\\bmidline\\b'
      - 'peripheral(ly)? inserted midline'""",
    "central_venous_catheter: exclude midline")

# ── 3. supplemental_oxygen: remove positive-pressure devices ──────────────
sub("""      - 'supplemental o2|supplemental oxygen'
      - 'nasal cannula|\\bnc\\b'
      - '\\bhfnc\\b|high[- ]flow'
      - 'non[- ]rebreather|\\bnrb\\b'
      - '\\bbipap\\b|\\bcpap\\b'
      - 'venturi mask|face mask oxygen'
    exclude: []
    notes: 'Bare `\\bnc\\b` is risky (also = no complaints). Review.'""",
    """      - 'supplemental o2|supplemental oxygen'
      - 'nasal cannula|nasal canula|\\bnc\\b'
      - '\\bhfnc\\b|high[- ]flow'
      - 'non[- ]rebreather|\\bnrb\\b'
      - 'venturi mask|face mask oxygen'
      - '\\bo2\\b (via|by|at)'
    exclude:
      - '\\bbipap\\b|\\bcpap\\b'
      - 'no complaints'
    notes: >
      CPAP and BiPAP removed after review: they deliver positive pressure,
      frequently on room air, so they are not supplemental oxygen under this
      feature's own definition. They were 1243 of 1525 occurrences (cpap 671,
      bipap 572), so this feature is now ~280 occurrences and genuinely means
      what it says. If non-invasive ventilation is wanted it should be its own
      feature -- a decision for the feature list, not this file.
      `nasal canula` is a corpus misspelling. Bare `\\bnc\\b` is retained but
      `no complaints` is excluded.""",
    "supplemental_oxygen: drop CPAP/BiPAP")

# ── 4. pneumonia: add PNA ─────────────────────────────────────────────────
sub("""      - 'pneumonia'
      - '\\bhcap\\b|\\bcap\\b|\\bvap\\b|\\bhap\\b'
      - 'lower respiratory.*infection'
      - 'bronchitis'
      - 'empyema'
    exclude:
      - 'pneumonitis'
    notes: 'Beware bare `cap` -- check the corpus for false positives.'""",
    """      - 'pneumonia'
      - '\\bpna\\b'
      - '\\bhcap\\b|\\bcap\\b|\\bvap\\b|\\bhap\\b'
      - 'lower respiratory.*infection'
      - 'bronchitis'
      - 'empyema'
    exclude:
      - 'pneumonitis'
    notes: >
      `pna` 798 occurrences added after review -- the standard abbreviation
      was completely unclaimed, roughly a 30% increase to this feature.
      Beware bare `cap` -- check the corpus for false positives.""",
    "pneumonia: add pna")

# ── 5. vancomycin: add the `vanc` clipping ────────────────────────────────
sub("""      - 'vancomycin|vanco\\b|vancocin'""",
    """      - 'vancomycin|vanco\\w*|\\bvanc\\b|vancocin'""",
    "vancomycin_glycopeptide: add vanc")

# ── 6. immunosuppressed_state: transplant acronyms ────────────────────────
sub("""      - '\\bgvhd\\b|graft[- ]versus[- ]host'
      - '\\bscid\\b|common variable immunodef'""",
    """      - '\\bgvhd\\b|graft[- ]versus[- ]host'
      - '\\bscid\\b|common variable immunodef'
      - '\\bolt\\b|\\bddkt\\b|\\blrkt\\b|\\bsot\\b'
      - '\\bbmt\\b|\\bhsct\\b|\\bsct\\b|stem cell transplant'""",
    "immunosuppressed_state: transplant acronyms")

# ── 7. surgery: cardiac procedure acronyms ────────────────────────────────
sub("""      - 'surgical site infection|\\bssi\\b'
      - 'post[- ]operative wound'""",
    """      - 'surgical site infection|\\bssi\\b'
      - 'post[- ]operative wound'
      - '\\bcabg\\b|\\bpci\\b|\\bavr\\b|\\bmvr\\b|\\btavr\\b'""",
    "surgery: cardiac acronyms")

p.write_text(src)
print(f"\n{edits} edits applied")
