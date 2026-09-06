"""Break down assertion disagreements by note type, label, and cue.

Reads the disagreement CSV written by ``score_assertions.py --errors-csv`` and
answers three questions that the aggregate per-axis F1 hides: whether
temporality fails on a particular note type, which entity strings drive the
experiencer false positives, and which gold hypothetical spans were missed
outright.

Used 2026-08-31 during the assertion tuning pass. The ``TYPE`` map is the 16
notes of the gold set, hardcoded because the note-type column lives in
``splits/manifest.csv``, which stays on the enclave. The input path is likewise
hardcoded to ``/tmp/assert_errors.csv``; pass ``--errors-csv`` to that path when
running ``score_assertions.py``.

RUN ON A MINERVA COMPUTE NODE. The disagreement CSV contains span text (PHI).
"""


import pandas as pd

TYPE = {
 '71778772':'Progress','110577574':'Progress','204449710':'Progress','110721936':'Progress',
 '136602524':'Consults','259491978':'Consults','141317604':'Consults','109752688':'Consults',
 '236578963':'H&P','259029479':'H&P','206602154':'H&P','109981187':'H&P',
 '244607651':'Discharge','259765625':'Discharge','146810273':'Discharge','110220141':'Discharge',
}
d = pd.read_csv('/tmp/assert_errors.csv', dtype={'note_id':str})
d['type'] = d.note_id.map(TYPE)
print('side values:', d.side.unique())

print('\n--- temporality errors by note type ---')
t = d[(d.axis=='temporality')]
print(pd.crosstab(t['type'], t.side).to_string())

print('\n--- temporality errors by label ---')
print(pd.crosstab(t.label, t.side).to_string())

print('\n--- experiencer FP texts (top 15) ---')
e = d[(d.axis=='experiencer') & (d.side=='pred_only')]
print(e.text.str.lower().value_counts().head(15).to_string())

print('\n--- hypothetical FN texts (all) ---')
h = d[(d.axis=='certainty') & (d.value=='hypothetical') & (d.side=='gold_only')]
print(h[['note_id','label','text']].to_string(index=False))
