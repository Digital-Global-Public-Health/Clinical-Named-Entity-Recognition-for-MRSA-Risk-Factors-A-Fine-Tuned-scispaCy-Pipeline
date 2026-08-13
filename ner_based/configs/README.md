# AIR-MS spaCy NER training

Run the wrapper from `ner_based/` with one of the nested learning-curve sizes
or the full training split:

```bash
bash scripts/train_ner.sh 2000
bash scripts/train_ner.sh 5000
bash scripts/train_ner.sh 10000
bash scripts/train_ner.sh full
```

Training uses CPU by default. Pass a GPU ID as the second argument when one is
available, for example `bash scripts/train_ner.sh full 0`.

The wrapper runs `spacy debug config`, `spacy debug data`, and `spacy train` in
that order. In the `debug data` output, misaligned entity spans should be close
to zero. A large count means the scispaCy tokenizer callback did not fire; do
not train until that is corrected.

The NER listener width is the verified `en_core_sci_sm` tok2vec width. Recheck
it in the target environment with:

```bash
python -c "import spacy; print(spacy.load('en_core_sci_sm').config['components']['tok2vec']['model']['encode']['width'])"
```

Training artifacts are written below `models/`. This directory must remain
gitignored because model checkpoints are large generated binaries.
