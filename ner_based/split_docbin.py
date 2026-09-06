"""Patient-level train/dev split of a pre-annotated DocBin.

Notes from one patient never cross the split boundary: clinical notes copy
forward heavily, so a note-level split leaks near-duplicate text into dev and
inflates measured performance. Patients are assigned greedily, largest first,
to whichever side is furthest below its target share of NOTES (not patients),
which keeps the note counts close even when chart sizes are very uneven.

Usage:
  python split_docbin.py IN.spacy OUT_DIR [--dev-frac 0.2] [--exclude FILE]
"""
import argparse, json, collections
from pathlib import Path

import spacy
from spacy.tokens import DocBin

ap = argparse.ArgumentParser()
ap.add_argument("docbin")
ap.add_argument("out_dir")
ap.add_argument("--dev-frac", type=float, default=0.2)
ap.add_argument("--exclude", help="file of PERSON_IDs to hold out entirely (e.g. gold set)")
args = ap.parse_args()

nlp = spacy.blank("en")
docs = list(DocBin().from_disk(args.docbin).get_docs(nlp.vocab))

excluded = set()
if args.exclude:
    excluded = {str(x).strip() for x in open(args.exclude) if x.strip()}

by_patient = collections.defaultdict(list)
held = []
for d in docs:
    pid = str(d.user_data.get("patient_id"))
    (held if pid in excluded else by_patient[pid]).append(d)

total = sum(len(v) for v in by_patient.values())
target_dev = total * args.dev_frac

train, dev = [], []
n_train = n_dev = 0
assign = {}
for pid, group in sorted(by_patient.items(), key=lambda kv: -len(kv[1])):
    # deficit = how far each side is below its target share
    if (target_dev - n_dev) > ((total - target_dev) - n_train):
        dev.extend(group); n_dev += len(group); assign[pid] = "dev"
    else:
        train.extend(group); n_train += len(group); assign[pid] = "train"

out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
for name, group in (("train", train), ("dev", dev), ("held_out", held)):
    if not group and name == "held_out":
        continue
    db = DocBin(store_user_data=True)
    for d in group:
        db.add(d)
    db.to_disk(out / f"{name}.spacy")

ents = lambda g: sum(len(d.ents) for d in g)
manifest = {
    "source_docbin": args.docbin,
    "dev_frac_target": args.dev_frac,
    "excluded_patients": sorted(excluded),
    "assignment": assign,
    "counts": {
        "train": {"patients": sum(v == "train" for v in assign.values()),
                  "notes": len(train), "entities": ents(train)},
        "dev": {"patients": sum(v == "dev" for v in assign.values()),
                "notes": len(dev), "entities": ents(dev)},
        "held_out": {"notes": len(held), "entities": ents(held)},
    },
}
(out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

# leakage assertion -- the whole point of the script
assert not ({p for p, v in assign.items() if v == "train"} &
            {p for p, v in assign.items() if v == "dev"})

c = manifest["counts"]
print(f"train: {c['train']['patients']} patients, {c['train']['notes']} notes, {c['train']['entities']} entities")
print(f"dev  : {c['dev']['patients']} patients, {c['dev']['notes']} notes, {c['dev']['entities']} entities")
print(f"dev note share: {len(dev)/total:.1%} (target {args.dev_frac:.0%})")
if held:
    print(f"held out: {len(held)} notes")
print("-> " + str(out))
