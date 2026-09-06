"""Measure how many gold spans originated as accepted recommender suggestions.

INCEpTION's recommenders were active while the six pilot notes were annotated
and switched off for the ten extension notes. If a large share of pilot gold
spans came from clicking "accept" on a machine suggestion, the pilot half is not
an independent standard and cannot be pooled with the extension half.

Joins the project export's ``learning_records`` (action ``ACCEPTED``) against the
gold span dump on (note_id, start, end, label), and separately reports how many
accepted spans were repeat occurrences of a text already annotated in that note.

Used 2026-08-29, when the gold set was extended from 6 to 16 notes. This is why
``eval_gold.py --split`` reports pilot and extension separately rather than only
the pooled figure.

Reads ``/tmp/gold_verify_spans.csv`` and the newest ``gold25_backup_*.zip`` in
the working directory. Neither is produced by anything in this repository; both
were enclave-local at the time.

RUN ON A MINERVA COMPUTE NODE. Reads span text (PHI).
"""


import json, zipfile, sys, glob
import pandas as pd
from collections import Counter

zp = sorted(glob.glob('gold25_backup_*.zip'))[-1]
d = json.loads(zipfile.ZipFile(zp).read('exportedproject.json'))
lr = pd.DataFrame(d['learning_records'])
lr['note_id'] = lr.document_name.str.replace('.txt', '', regex=False)

acc = lr[lr.action == 'ACCEPTED']
spans = pd.read_csv('/tmp/gold_verify_spans.csv', dtype={'note_id': str})

key = ['note_id', 'start', 'end', 'label']
a = acc.rename(columns={'begin': 'start'})[['note_id', 'start', 'end', 'label']].drop_duplicates()
a['note_id'] = a.note_id.astype(str)

m = spans.merge(a, on=key, how='left', indicator=True)
n_match = (m._merge == 'both').sum()

print(f'gold spans in dump      {len(spans)}')
print(f'ACCEPTED records        {len(acc)}  (unique offsets: {len(a)})')
print(f'gold spans matching     {n_match}   ({n_match/len(spans):.1%})')
print(f'gold spans hand-made    {len(spans)-n_match}')
print()
print('accepted-by-label:', dict(Counter(acc.label)))
print('accepted-by-note :', dict(Counter(acc.note_id)))
print()

# how many accepted spans are repeats of a text already annotated in that note?
acc = acc.sort_values('timestamp')
first = ~acc.duplicated(['note_id', 'text', 'label'])
print(f'accepted, first occurrence of that text in note   {first.sum()}')
print(f'accepted, repeat occurrence of an existing text   {(~first).sum()}')
print()
print('top repeated accepted texts:')
rep = acc[~first].groupby(['text', 'label']).size().sort_values(ascending=False)
print(rep.head(12).to_string())
