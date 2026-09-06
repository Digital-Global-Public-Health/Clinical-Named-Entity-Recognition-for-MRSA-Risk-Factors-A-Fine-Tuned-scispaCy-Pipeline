"""Compare two teacher models' pre-annotations span-for-span.

Run before committing GPU time to pre-annotating the full corpus: how much do
``llama3.1:70b`` and ``gemma3:27b`` actually disagree? Reports per note the
spans both found, the spans unique to each, and their Jaccard overlap, matching
on (lowercased text, label) rather than offsets so tokenization differences do
not count as disagreement.

Used 2026-08-10. The llama family was kept. ``procedure_gap.py`` is the same
comparison narrowed to PROCEDURE, the label the two models diverged on most.

The ``llama3.1:70b`` named here is this comparison's model, not the production
teacher: the run that produced ``annotations/batch01_v2``, ``batch02`` and
``batch03`` used ``llama3.3:70b``.

Both smoke-run directories it reads are enclave-local and gitignored.
"""


import json
from pathlib import Path

A = Path('annotations/preannotations_smoke/verified')          # llama3.1:70b
B = Path('annotations/preannotations_smoke_gemma3/verified')   # gemma3:27b

def spans(p):
    d = json.load(open(p))
    return {(s['text'].strip().lower(), s['label']) for s in d['spans']}

ta = tb = tboth = 0
for f in sorted(B.glob('*.json')):
    if not (A / f.name).exists():
        print(f"{f.stem}  no llama counterpart"); continue
    a, b = spans(A / f.name), spans(f)
    j = len(a & b) / len(a | b)
    ta += len(a); tb += len(b); tboth += len(a & b)
    print(f"{f.stem}  both={len(a&b):3d}  llama_only={len(a-b):3d}  gemma_only={len(b-a):3d}  jaccard={j:.2f}")
print(f"\ntotals: llama={ta} gemma={tb} shared={tboth}")
