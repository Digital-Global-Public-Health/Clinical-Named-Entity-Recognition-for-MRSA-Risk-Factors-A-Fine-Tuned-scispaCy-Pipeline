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
