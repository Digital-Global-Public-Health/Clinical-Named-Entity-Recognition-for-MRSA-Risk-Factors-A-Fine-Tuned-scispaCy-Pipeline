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
