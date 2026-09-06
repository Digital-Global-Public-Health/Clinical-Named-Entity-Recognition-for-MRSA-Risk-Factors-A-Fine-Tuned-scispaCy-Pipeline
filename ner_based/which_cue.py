"""Attribute each hypothetical false positive to the ConText cue that caused it.

Gold spans are injected directly as ``doc.ents`` and every pipe except ``ner``
is run over them, so this measures the cue set in isolation from NER recall.
For each entity the layer marks ``is_hypothetical`` where gold does not, the
literal of every modifier attached to it is counted.

Used 2026-08-31. The counts identified which cues were over-scoping and informed
the backward scope caps now on the hypothetical rules in ``src/ner/assertion.py``.
``which_cue_exp.py`` is the same script for the experiencer axis.

Expects ``/tmp/gold6_spans.csv`` from ``project6_to_spans.py`` and one
``<note_id>.txt`` per gold note under ``/tmp/gold16_txt``; neither is produced
by anything in this repository -- both were enclave scratch files.

RUN ON A MINERVA COMPUTE NODE. Reads note text (PHI).
"""


import pandas as pd, spacy
from src.ner.assertion import build_assertion_pipeline

TXT = '/tmp/gold16_txt'
s = pd.read_csv('/tmp/gold6_spans.csv', dtype={'note_id': str})
nlp = build_assertion_pipeline('models/ner_full/model-best')

from collections import Counter
cues = Counter()
for note, grp in s.groupby('note_id'):
    text = open(f'{TXT}/{note}.txt').read()
    doc = nlp.make_doc(text)
    ents = []
    for _, r in grp.iterrows():
        sp = doc.char_span(int(r.start), int(r.end), label=r.value,
                           alignment_mode='expand')
        if sp is not None:
            ents.append(sp)
    try:
        doc.ents = spacy.util.filter_spans(ents)
    except Exception:
        continue
    for name, pipe in nlp.pipeline:
        if name != 'ner':
            doc = pipe(doc)
    gold = {(int(r.start), int(r.end)): r.certainty for _, r in grp.iterrows()}
    for e in doc.ents:
        if getattr(e._, 'is_hypothetical', False):
            g = gold.get((e.start_char, e.end_char))
            if g != 'hypothetical':
                for m in e._.modifiers:
                    cues[m.rule.literal] += 1
print(cues.most_common())
