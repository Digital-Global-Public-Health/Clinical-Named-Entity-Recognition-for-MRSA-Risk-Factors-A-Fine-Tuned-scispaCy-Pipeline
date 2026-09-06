"""Attribute each experiencer false positive to the ConText cue that caused it.

The experiencer counterpart of ``which_cue.py``: for every entity the layer
marks ``is_family`` where gold does not, count the literal of each attached
modifier. It differs from ``which_cue.py`` in skipping ``tok2vec`` as well as
``ner`` when replaying the pipeline, and in printing the note id when a span set
cannot be filtered rather than silently continuing.

Used 2026-08-31. This is what identified medspaCy's bare ``family`` cue as the
dominant source -- it fires on provenance lines such as "Source: Patient, Family
and Team" and then scopes forward across the flattened export into unrelated
sections. The rule is now disabled via ``PACKAGED_RULE_FIXES`` in
``src/ner/assertion.py``.

Same enclave-scratch inputs as ``which_cue.py``.

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
    except Exception as ex:
        print('SKIP', note, ex); continue
    for name, pipe in nlp.pipeline:
        if name not in ('ner', 'tok2vec'):
            doc = pipe(doc)
    gold = {(int(r.start), int(r.end)): r.experiencer for _, r in grp.iterrows()}
    for e in doc.ents:
        if getattr(e._, 'is_family', False):
            g = gold.get((e.start_char, e.end_char))
            if g != 'other':
                for m in e._.modifiers:
                    cues[m.rule.literal] += 1
print(cues.most_common())
