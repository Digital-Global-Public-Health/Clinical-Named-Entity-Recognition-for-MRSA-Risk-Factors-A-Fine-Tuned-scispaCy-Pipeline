"""Allow PROCEDURE spans to match the immunosuppressed_state feature.

A follow-up to ``patch_lexicon_02.py`` from the same 2026-08-26 corpus review.
That patch added the transplant acronyms ``olt`` and ``ddkt`` to
``immunosuppressed_state``, but the feature was restricted to
``entity_types: [DISEASE]`` -- and the NER model tags a transplant as a
PROCEDURE, so the newly added acronyms could never match. Widens the feature to
accept both labels.

Applied once; the change is already in ``lexicon.yaml``. Aborts if its anchor
does not match exactly once, so re-running is safe.

Run from ner_based/ with lexicon.yaml present.
"""


import pathlib, sys
p = pathlib.Path("lexicon.yaml"); src = p.read_text()
old = """  - name: immunosuppressed_state
    label: Transplant / immunosuppressed state
    entity_types: [DISEASE]"""
new = """  - name: immunosuppressed_state
    label: Transplant / immunosuppressed state
    entity_types: [DISEASE, PROCEDURE]"""
if src.count(old) != 1:
    sys.exit(f"anchor matched {src.count(old)} times")
p.write_text(src.replace(old, new))
print("ok  immunosuppressed_state: allow PROCEDURE spans")
