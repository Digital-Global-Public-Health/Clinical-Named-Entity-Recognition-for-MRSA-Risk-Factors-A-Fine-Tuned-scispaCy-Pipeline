"""Re-derive span offsets from existing verified JSONs, without calling the LLM.

Reconstructs each note's proposal list from `spans` (kept) + `rejected[].proposed`
(dropped), then replays it through verify_model_response so the fixed
word-boundary matcher applies. Model output is held constant; only offset
logic changes.
"""
import json, sys, collections
from pathlib import Path
from types import SimpleNamespace

from src.ner.preannotate import verify_model_response

src_dir = Path(sys.argv[1])          # annotations/batch01/verified
out_dir = Path(sys.argv[2])          # annotations/batch01_rv/verified
out_dir.mkdir(parents=True, exist_ok=True)

before = collections.Counter()
after = collections.Counter()
reasons = collections.Counter()
files = sorted(src_dir.glob("*.json"))

for i, f in enumerate(files, 1):
    d = json.load(open(f))

    entities, seen = [], set()
    for s in d.get("spans", []):                     # kept proposals
        key = (s["text"], s["label"])
        if key in seen:
            continue                                 # collapse ambiguous duplicates
        seen.add(key)
        entities.append({"text": s["text"], "label": s["label"]})
    for r in d.get("rejected", []):                  # dropped proposals
        entities.append(dict(r["proposed"]))

    note = SimpleNamespace(
        note_id=d["note_id"],
        patient_id=d.get("patient_id"),
        text=d["text"],
    )
    result = verify_model_response(note, {"entities": entities})
    out = result.to_json_dict()
    (out_dir / f.name).write_text(json.dumps(out, indent=2) + "\n")

    for s in d.get("spans", []):
        before[s["label"]] += 1
    for s in out.get("spans", []):
        after[s["label"]] += 1
    for r in out.get("rejected", []):
        reasons[r.get("reason")] += 1

    if i % 500 == 0:
        print(f"{i}/{len(files)}", flush=True)

print("\nnotes:", len(files))
print("BEFORE:", dict(before))
print("AFTER :", dict(after))
print("REASONS:", dict(reasons))
