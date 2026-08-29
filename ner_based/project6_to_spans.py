import zipfile, glob, re, csv, sys
from collections import defaultdict, Counter

import os
zp = max(glob.glob('goldassert_*.zip'), key=os.path.getmtime)
z = zipfile.ZipFile(zp)
names = [n for n in z.namelist()
         if n.startswith('annotation/') and n.endswith('/admin.tsv')]

rows, per_doc = [], {}
for n in sorted(names):
    note = n.split('/')[1].replace('.tsv', '')
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
