"""Parse the INCEpTION project 6 export into a gold assertion span CSV.

Project 6 holds the gold spans re-typed onto ``webanno.custom.Assertion`` by
``rewrite_tsv_layer.py``, so each token row carries six feature columns rather
than the built-in layer's one. This reads the column names from the ``#T_SP=``
header instead of assuming positions, merges multi-token spans by the ``[n]``
disambiguation id shared across their rows, and emits one row per span with all
five attribute values.

Used 2026-08-29, after the assertion annotation pass. The output is the gold
attribute set that ``which_cue.py`` and ``which_cue_exp.py`` score against;
``score_assertions.py`` parses the same export independently and does not read
this file.

RUN ON A MINERVA COMPUTE NODE. The output CSV contains span text (PHI) and is
written to /tmp deliberately, so it is never inside the repository.
"""


import zipfile, glob, re, csv, sys
from collections import defaultdict, Counter

import os
zp = max(glob.glob('goldassert_*.zip'), key=os.path.getmtime)
z = zipfile.ZipFile(zp)
names = [n for n in z.namelist()
         if n.startswith('annotation/') and n.endswith('/admin.tsv')]

rows, per_doc = [], {}
for n in sorted(names):
    note = os.path.splitext(n.split('/')[1])[0]
    feats, groups, auto = None, defaultdict(list), 0
    for line in z.read(n).decode('utf-8').splitlines():
        if line.startswith('#T_SP='):
            feats = line.split('|')[1:]
            continue
        if not line or line.startswith('#'):
            continue
        p = line.split('\t')
        if len(p) < 3 + len(feats):
            continue
        beg, end = p[1].split('-')
        cols = p[3:3 + len(feats)]
        if cols[-1] == '_':
            continue
        n_ann = max(len(c.split('|')) for c in cols)
        for i in range(n_ann):
            vals = [c.split('|')[i] if i < len(c.split('|')) else c.split('|')[0]
                    for c in cols]
            m = re.search(r'\[(\d+)\]$', vals[-1])
            gid = m.group(1) if m else f'auto{auto}'
            if not m:
                auto += 1
            vals = [re.sub(r'\[\d+\]$', '', v) for v in vals]
            groups[(note, gid)].append((int(beg), int(end), p[2], vals))

    for (note_, gid), toks in groups.items():
        toks.sort()
        v = toks[0][3]
        rows.append(dict(note_id=note_, start=toks[0][0], end=toks[-1][1],
                         text=' '.join(t[2] for t in toks),
                         **dict(zip(feats, v))))
    per_doc[note] = sum(1 for r in rows if r['note_id'] == note)

print(f'zip: {zp}')
print(f'documents: {len(names)} | spans: {len(rows)}')
print('per doc:', per_doc)
print('labels :', dict(Counter(r['value'] for r in rows)))
for f in ['polarity', 'certainty', 'temporality', 'experiencer', 'allergy']:
    print(f'{f:12}', dict(Counter(r[f] for r in rows)))

with open('/tmp/gold6_spans.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('wrote /tmp/gold6_spans.csv  -- CONTAINS PHI, keep on Minerva')
