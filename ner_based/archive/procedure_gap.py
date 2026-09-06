"""Compare two teacher models on PROCEDURE spans specifically.

The label the two candidate teachers disagreed on most, and the one that matters
most here: indwelling devices and lines are the primary MRSA acquisition route,
so a teacher that systematically misses them caps the student's recall on the
features the thesis is about.

Unlike ``compare_runs.py``, this matches on offset overlap rather than exact
text, and de-duplicates by span text so a term repeated through a note does not
pad the lists. Prints, per note, what both models found, what gemma missed, and
what only gemma found.

Used 2026-08-10, alongside ``compare_runs.py``. The llama family was kept; the
model named there is this comparison's, not the production teacher, which was
``llama3.3:70b``.

Both smoke-run directories it reads are enclave-local and gitignored.
"""


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
