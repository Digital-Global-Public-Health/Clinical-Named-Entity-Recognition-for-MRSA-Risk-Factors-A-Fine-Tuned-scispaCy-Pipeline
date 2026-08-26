#!/usr/bin/env python
"""Build the MRSA NER feature matrix.

Runs the assertion pipeline over the corpus, gates every entity on the four
universal assertion axes, matches surviving entities to the 48 features in
lexicon.yaml, and emits a matrix at note, visit or patient level.

DESIGN NOTES

Assertion gating. Four axes are universal gates -- an entity failing any of
them is not evidence that the thing occurred:
    is_negated       "no cellulitis"                  -> drop
    is_family        "mother had breast cancer"       -> drop
    is_hypothetical  "monitor for sedation"           -> drop
    is_allergy       drug in an allergy list          -> drop (MEDICATION)
`is_uncertain` is deliberately NOT a gate: "possible pneumonia" is weak
evidence but it is evidence, and these features are sparse. Uncertain
mentions are counted separately so the sensitivity can be reported.

Temporality is NOT a gate. It scores 38.7 F1 (26-08 measurement), so gating on
it would silently zero out most true positives. Instead each feature carries a
companion count of how many of its supporting mentions were marked historical,
from which an `_all_hist` flag is derived at any aggregation level. This
preserves the distinction without betting the feature value on an unreliable
axis. Report it; do not enforce it.

Storage. The intermediate CSV stores four counts per feature -- n, n_hist,
n_uncertain, n_dropped -- rather than booleans, because counts roll up by
summation to any level and booleans do not. `--level` derives the binary view.

Counts vs binaries. Binary flags are the primary feature. Raw counts are
emitted too, but note that documentation volume differs sharply between label
groups in this cohort (cases ~13 notes/visit, controls ~2.7), so counts partly
measure how much was written. Normalise or prefer binaries.

RUN ON A MINERVA COMPUTE NODE. Input is note text (PHI). The output matrix
contains no text, but does contain note/patient identifiers.

Usage:
    python build_feature_matrix.py --limit 200 --out /tmp/fm_test.csv
    python build_feature_matrix.py --out outputs/feature_matrix_note.csv
    python build_feature_matrix.py --from-counts outputs/feature_matrix_note.csv \\
        --level visit --out outputs/feature_matrix_visit.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PARQUET = ("/sc/arion/work/rademt02/airms_notes/extracted/"
           "cohort_all_notes.parquet")
KEYS = ["NOTE_ID", "PERSON_ID", "VISIT_OCCURRENCE_ID", "NOTE_DATETIME",
        "NOTE_TITLE", "LABEL"]
SUFFIXES = ["n", "nh", "nu", "nd"]   # total, historical, uncertain, dropped

# Axes that gate. An entity failing any of these is not evidence.
GATES = ["is_negated", "is_family", "is_hypothetical", "is_allergy"]


def load_lexicon(path):
    import yaml
    with open(path) as fh:
        feats = yaml.safe_load(fh)["features"]
    by_name = {f["name"]: f for f in feats}
    ref = re.compile(r"^__ref__:(.+)$")

    def expand(name, seen):
        if name in seen:
            sys.exit(f"circular __ref__ at {name}")
        out = []
        for pat in by_name[name].get("include") or []:
            m = ref.match(pat)
            if m:
                out.extend(expand(m.group(1), seen | {name}))
            else:
                out.append(pat)
        return out

    for f in feats:
        f["_inc"] = [re.compile(p, re.I) for p in expand(f["name"], set())]
        f["_exc"] = [re.compile(p, re.I)
                     for p in (f.get("exclude") or [])]
        f["_types"] = set(f.get("entity_types")
                          or ["DISEASE", "MEDICATION", "PROCEDURE"])
    return feats


def matches(feat, label, text):
    if label not in feat["_types"]:
        return False
    if not any(p.search(text) for p in feat["_inc"]):
        return False
    return not any(p.search(text) for p in feat["_exc"])


def counts_as_evidence(ent):
    """The universal assertion gate. See module docstring."""
    from spacy.tokens import Span
    for attr in GATES:
        if Span.has_extension(attr) and getattr(ent._, attr, False):
            return False
    return True


def build(args):
    import pandas as pd
    from src.ner.assertion import build_assertion_pipeline

    feats = load_lexicon(args.lexicon)
    names = [f["name"] for f in feats]
    print(f"lexicon: {len(feats)} features", file=sys.stderr)

    df = pd.read_parquet(args.parquet)
    if args.limit:
        df = df.head(args.limit)
    print(f"notes: {len(df)}", file=sys.stderr)

    nlp = build_assertion_pipeline(args.model)
    print(f"pipes: {nlp.pipe_names}", file=sys.stderr)

    texts = df["NOTE_TEXT"].fillna("").tolist()
    meta = df[[k for k in KEYS if k in df.columns]].to_dict("records")

    rows = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=args.batch_size)):
        rec = dict(meta[i])
        rec["n_chars"] = len(doc.text)
        rec["n_entities"] = len(doc.ents)
        tally = {n: [0, 0, 0, 0] for n in names}
        for ent in doc.ents:
            txt = ent.text.lower().strip()
            keep = counts_as_evidence(ent)
            hist = bool(getattr(ent._, "is_historical", False))
            unc = bool(getattr(ent._, "is_uncertain", False))
            for f in feats:
                if not matches(f, ent.label_, txt):
                    continue
                t = tally[f["name"]]
                if not keep:
                    t[3] += 1
                    continue
                t[0] += 1
                t[1] += hist
                t[2] += unc
        for n in names:
            for s, v in zip(SUFFIXES, tally[n]):
                rec[f"{n}_{s}"] = v
        rows.append(rec)
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i+1}/{len(texts)}", file=sys.stderr)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(out)} rows x {len(out.columns)} cols",
          file=sys.stderr)
    summarise(out, names)


def derive(counts, names, level, keys_present):
    """Roll counts up to `level` and derive the binary view."""
    import pandas as pd

    key = {"note": "NOTE_ID", "visit": "VISIT_OCCURRENCE_ID",
           "patient": "PERSON_ID"}[level]
    if key not in keys_present:
        sys.exit(f"{key} not in the counts file")

    agg = {f"{n}_{s}": "sum" for n in names for s in SUFFIXES}
    agg["n_entities"] = "sum"
    agg["n_chars"] = "sum"
    agg["NOTE_ID"] = "size"
    for k in ("PERSON_ID", "LABEL", "VISIT_OCCURRENCE_ID"):
        if k in keys_present and k != key:
            agg[k] = "first"
    if level == "note":
        g = counts.set_index(key)
        g["n_notes"] = 1
    else:
        g = counts.groupby(key).agg(agg).rename(columns={"NOTE_ID": "n_notes"})

    # Build all columns at once; assigning ~250 columns one by one fragments
    # the frame and emits a PerformanceWarning per column.
    cols = {}
    for k in ("PERSON_ID", "VISIT_OCCURRENCE_ID", "LABEL"):
        if k in g.columns:
            cols[k] = g[k]
    cols["n_notes"] = g["n_notes"]
    cols["n_entities"] = g["n_entities"]
    for n in names:
        tot = g[f"{n}_n"]
        cols[n] = (tot > 0).astype(int)
        cols[f"{n}_count"] = tot
        cols[f"{n}_per_note"] = (tot / g["n_notes"]).round(3)
        cols[f"{n}_all_hist"] = ((tot > 0) & (g[f"{n}_nh"] == tot)).astype(int)
        cols[f"{n}_any_uncertain"] = ((tot > 0) & (g[f"{n}_nu"] > 0)).astype(int)
    return pd.DataFrame(cols, index=g.index).reset_index()


def summarise(counts, names):
    print(f"\n{'feature':<30} {'rows>0':>8} {'%':>7} {'mentions':>9} "
          f"{'dropped':>8} {'all_hist':>9}")
    for n in names:
        tot = counts[f"{n}_n"]
        pos = (tot > 0).sum()
        ah = ((tot > 0) & (counts[f"{n}_nh"] == tot)).sum()
        print(f"{n:<30} {pos:>8} {100*pos/len(counts):>6.1f}% "
              f"{tot.sum():>9} {counts[f'{n}_nd'].sum():>8} {ah:>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="lexicon.yaml")
    ap.add_argument("--model", default="models/ner_full/model-best")
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--progress", type=int, default=500)
    ap.add_argument("--from-counts", default=None,
                    help="skip inference; roll up an existing counts CSV")
    ap.add_argument("--level", default=None,
                    choices=["note", "visit", "patient"])
    args = ap.parse_args()

    if args.from_counts:
        import pandas as pd
        feats = load_lexicon(args.lexicon)
        names = [f["name"] for f in feats]
        counts = pd.read_csv(args.from_counts)
        level = args.level or "note"
        out = derive(counts, names, level, set(counts.columns))
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}: {len(out)} rows x {len(out.columns)} cols "
              f"({level} level)", file=sys.stderr)
        return

    build(args)


if __name__ == "__main__":
    main()
