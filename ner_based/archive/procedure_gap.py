import json
from pathlib import Path

A = Path('annotations/preannotations_smoke/verified')          # llama3.1:70b
B = Path('annotations/preannotations_smoke_gemma3/verified')   # gemma3:27b

def procs(p):
    return [s for s in json.load(open(p))['spans'] if s['label'] == 'PROCEDURE']

def hits(s, others):
    return [o for o in others if s['start_char'] < o['end_char'] and o['start_char'] < s['end_char']]

for f in sorted(B.glob('*.json')):
    a, b = procs(A / f.name), procs(f)
    # de-dup by text so repeated occurrences don't pad the list
    missed = sorted({s['text'].strip() for s in a if not hits(s, b)}, key=str.lower)
    extra  = sorted({s['text'].strip() for s in b if not hits(s, a)}, key=str.lower)
    caught = sorted({s['text'].strip() for s in a if hits(s, b)}, key=str.lower)
    print(f"\n=== {f.stem}  llama={len(a)} gemma={len(b)} ===")
    print(f"  BOTH FOUND ({len(caught)}): {caught}")
    print(f"  GEMMA MISSED ({len(missed)}): {missed}")
    print(f"  GEMMA ONLY ({len(extra)}): {extra}")
