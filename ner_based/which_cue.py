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
