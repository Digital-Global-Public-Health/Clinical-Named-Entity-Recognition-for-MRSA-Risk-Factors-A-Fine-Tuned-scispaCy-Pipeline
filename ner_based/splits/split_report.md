# Split report

- seed: 7
- targets: {'train': 0.7, 'dev': 0.15, 'test': 0.15}
- tokenizer: en_core_sci_sm
- min_chars: 200
- notes: 18669 | patients: 50 | entities: 571455

## Notes and entities per split

| split | patients | notes | % | entities |
|---|---|---|---|---|
| train | 24 | 13071 | 70.0% | 396939 |
| dev | 13 | 2793 | 15.0% | 87657 |
| test | 13 | 2805 | 15.0% | 86859 |

## Note type x split (notes / entities)

| type | train | dev | test |
|---|---|---|---|
| Consults | 890 / 45268 | 171 / 7199 | 187 / 7605 |
| Discharge Summary | 284 / 11048 | 51 / 2171 | 60 / 2192 |
| H&P | 316 / 20268 | 68 / 4257 | 85 / 4601 |
| Progress Notes | 11581 / 320355 | 2503 / 74030 | 2473 / 72461 |

## Label balance

| split | 0 | 1 |
|---|---|---|
| train | 7961 | 5110 |
| dev | 1308 | 1485 |
| test | 871 | 1934 |

## Build statistics

- dropped_overlapping: 14500
- entities: 571455
- skipped_invalid_label: 1304

Note: `alignment_mode='expand'` snaps model character offsets to token boundaries rather than discarding the span; `dropped_overlapping` are spans spaCy cannot store in `doc.ents`.
