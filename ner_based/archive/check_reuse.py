"""Plan the gold-set extension: what can be reused, and from which patients.

The gold set was extended from the six pilot notes to sixteen, structured as
four test-split patients x four note types. Annotating four fresh patients would
have discarded the pilot work, so this reports which of the six existing notes
fall inside a candidate patient x type grid, which test patients carry all four
note types at all, and which triple of patients maximises reuse.

Used 2026-08-29. Its output chose the patients that ``draw_new_gold.py`` then
drew from.

Patient identifiers are pseudonymised to P01..Pnn before printing, so no
PERSON_ID reaches stdout. Reads ``splits/manifest.csv``, which stays on the
enclave.
"""


import pandas as pd
from itertools import combinations

m = pd.read_csv('splits/manifest.csv', dtype={'note_id':str,'person_id':str})
test = m[m.split == 'test']

DONE = ['109981187','110220141','110577574','110721936','136602524','141317604']
TYPES = ['Progress Notes','Consults','H&P','Discharge Summary']

# pseudonyms so no person_id is ever printed
pmap = {p: f'P{i:02d}' for i, p in enumerate(sorted(test.person_id.unique()), 1)}

d = test[test.note_id.isin(DONE)].copy()
d['P'] = d.person_id.map(pmap)
print('--- the six annotated notes ---')
print(d[['note_id','P','note_type','n_chars','n_entities']]
        .sort_values(['P','note_type']).to_string(index=False))
missing = set(DONE) - set(d.note_id)
if missing:
    print('NOT IN TEST SPLIT:', missing)
print()

avail = (test.groupby(['person_id','note_type']).size()
             .unstack(fill_value=0).reindex(columns=TYPES, fill_value=0))
avail.index = [pmap[p] for p in avail.index]
print('--- notes available per test patient, by type ---')
print(avail.to_string())
print()

done_cells = set(zip(d.P, d.note_type))
full = [p for p in avail.index if (avail.loc[p, TYPES] > 0).all()]
print(f'patients with all four types: {len(full)} of {len(avail)}')
print()

rows = sorted(((sum((p,t) in done_cells for p in c for t in TYPES), c)
               for c in combinations(full, 3)), reverse=True)
best = rows[0][0]
print('--- best triples ---')
for reuse, combo in rows:
    if reuse < best:
        break
    have = [f'{p}:{t}' for p in combo for t in TYPES if (p,t) in done_cells]
    print(f'reuse {reuse}/12 | new notes needed {12-reuse} | {combo} | have: {have}')
