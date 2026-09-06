"""Draw the ten extension notes for the 16-note gold set.

Takes the patients ``check_reuse.py`` selected, and for every (patient, note
type) cell not already covered by one of the six pilot notes, draws one note at
random from the test split. Seed 7, sorted candidate pool, so the draw is
reproducible; a cell with only one candidate is drawn anyway and flagged as
forced.

Used 2026-08-29. Wrote ``splits/new_gold_10.csv``, the list that was then
annotated in INCEpTION and became the extension half of the gold set.

Patient identifiers are pseudonymised to P01..Pnn before printing. Reads
``splits/manifest.csv``, which stays on the enclave.
"""


import pandas as pd
import numpy as np

SEED  = 7
KEEP  = ['P02', 'P03', 'P08', 'P09']
TYPES = ['Progress Notes', 'Consults', 'H&P', 'Discharge Summary']
DONE  = ['109981187', '110220141', '110577574',
         '110721936', '136602524', '141317604']

m = pd.read_csv('splits/manifest.csv', dtype={'note_id': str, 'person_id': str})
test = m[m.split == 'test'].copy()
pmap = {p: f'P{i:02d}' for i, p in enumerate(sorted(test.person_id.unique()), 1)}
test['P'] = test.person_id.map(pmap)

have = set(zip(test[test.note_id.isin(DONE)].P,
               test[test.note_id.isin(DONE)].note_type))
pool = test[~test.note_id.isin(DONE)]

rng, picks = np.random.default_rng(SEED), []
for p in KEEP:
    for t in TYPES:
        if (p, t) in have:
            continue
        c = pool[(pool.P == p) & (pool.note_type == t)].sort_values('note_id')
        if c.empty:
            print(f'!! no candidate for {p} / {t}')
            continue
        if len(c) == 1:
            print(f'   note: {p} / {t} drawn from a pool of 1 (forced)')
        picks.append(c.iloc[[rng.integers(len(c))]])

new = pd.concat(picks)
print()
print(new[['note_id', 'P', 'note_type', 'n_chars', 'n_entities']].to_string(index=False))
print(f'\n{len(new)} new notes | {int(new.n_chars.sum()):,} chars | '
      f'{int(new.n_entities.sum())} silver entities')

new[['note_id', 'P', 'note_type', 'n_chars']].to_csv('splits/new_gold_10.csv', index=False)
print('wrote splits/new_gold_10.csv')
