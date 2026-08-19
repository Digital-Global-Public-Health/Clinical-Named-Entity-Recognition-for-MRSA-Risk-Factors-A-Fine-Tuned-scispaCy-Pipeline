"""Compare ConText with and without the AIR.MS cue list.

Both configurations keep medspaCy's packaged default rules AND the structural
TERMINATE rules (space runs, bullets), since those repair the flattened export
and are not part of what is being ablated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import medspacy
from spacy.tokens import DocBin

from src.ner.assertion import AIRMS_CONTEXT_RULES, ALLOWED_LABELS, _load_custom_code

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = PROJECT_ROOT / "models/ner_10000/model-best"
GOLD = PROJECT_ROOT / "annotations/gold_export/gold.spacy"
FLAGS = ["is_negated", "is_historical", "is_hypothetical", "is_uncertain", "is_family"]

TERMINATORS = [r for r in AIRMS_CONTEXT_RULES if r.category == "TERMINATE"]
CUES = [r for r in AIRMS_CONTEXT_RULES if r.category != "TERMINATE"]


def build(rules):
    nlp = medspacy.load(str(MODEL), medspacy_enable=["medspacy_context"], load_rules=True)
    nlp.add_pipe("medspacy_pyrush", after="ner")
    nlp.get_pipe("medspacy_context").add(rules)
    return nlp


def tally(nlp, texts):
    counts = dict.fromkeys(FLAGS, 0)
    total = 0
    for doc in nlp.pipe(texts):
        for ent in doc.ents:
            if ent.label_ not in ALLOWED_LABELS:
                continue
            total += 1
            for f in FLAGS:
                if getattr(ent._, f, False):
                    counts[f] += 1
    return total, counts


_load_custom_code()
print(f"AIR.MS rules: {len(CUES)} cues + {len(TERMINATORS)} terminators")

base = build(TERMINATORS)
texts = [d.text for d in DocBin().from_disk(GOLD).get_docs(base.vocab)]

n_off, off = tally(base, texts)
n_on, on = tally(build(TERMINATORS + CUES), texts)

print(f"\nentities: defaults-only {n_off} | with AIR.MS cues {n_on}\n")
print(f"{'flag':<18}{'defaults':>10}{'+AIR.MS':>10}{'delta':>8}")
for f in FLAGS:
    print(f"{f:<18}{off[f]:>10}{on[f]:>10}{on[f] - off[f]:>+8}")
