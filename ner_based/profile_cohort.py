#!/usr/bin/env python
"""
Profile the AIR-MS cohort before designing the train/val/test split.

The split must satisfy several constraints at once:
  - patients are indivisible (copy-forward means notes from one patient must
    never straddle splits)
  - roughly 70/15/15 by NOTE count
  - all note types represented in every split, especially test
  - LABEL (case/control) balance preserved

With ~50 patients those constraints can conflict. This script reports whether a
feasible split exists before any splitting code is written.

Usage
  python profile_cohort.py --input /sc/arion/work/rademt02/airms_notes/extracted/cohort_all_notes.parquet
"""

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="NOTE_TEXT")
    ap.add_argument("--id-col", default="NOTE_ID")
    ap.add_argument("--patient-col", default="PERSON_ID")
    ap.add_argument("--type-col", default="NOTE_TITLE")
    ap.add_argument("--label-col", default="LABEL")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    df["_len"] = df[args.text_col].str.len()
    kept = df[df._len >= args.min_chars]

    P, T, L = args.patient_col, args.type_col, args.label_col

    print("=" * 70)
    print("COHORT PROFILE")
    print("=" * 70)
    print(f"notes total            : {len(df)}")
    print(f"notes >= {args.min_chars} chars     : {len(kept)}")
    print(f"patients               : {df[P].nunique()}")
    print(f"note types             : {df[T].nunique()}")
    print()

    # ---- note types --------------------------------------------------------
    print("-- notes per type (>= min_chars) --")
    vc = kept[T].value_counts()
    for t, c in vc.items():
        print(f"  {str(t)[:45]:<45} {c:>7}  ({100*c/len(kept):>5.1f}%)")
    print()

    # ---- label -------------------------------------------------------------
    if L in df.columns:
        print("-- label balance --")
        print("  by note   :", kept[L].value_counts().to_dict())
        pl = kept.groupby(P)[L].agg(lambda s: s.mode().iat[0])
        print("  by patient:", pl.value_counts().to_dict())
        print()

    # ---- notes per patient -------------------------------------------------
    per_pat = kept.groupby(P).size().sort_values(ascending=False)
    print("-- notes per patient --")
    print(f"  min {per_pat.min()} | median {int(per_pat.median())} | "
          f"mean {per_pat.mean():.0f} | max {per_pat.max()}")
    print(f"  top 10 patients hold {100*per_pat.head(10).sum()/len(kept):.1f}% of notes")
    print(f"  largest single patient is {100*per_pat.max()/len(kept):.1f}% of the corpus")
    print("  distribution:", per_pat.head(15).tolist(), "...")
    print()

    # ---- THE feasibility question -----------------------------------------
    # A note type can only reach the test split if enough patients carry it.
    print("-- note-type coverage across patients (feasibility) --")
    cov = kept.groupby(T)[P].nunique().sort_values()
    n_pat = kept[P].nunique()
    target_test_patients = max(1, round(n_pat * args.test_frac))
    print(f"  (a ~{args.test_frac:.0%} test split is about "
          f"{target_test_patients} patients of {n_pat})")
    print()
    for t, k in cov.items():
        # probability a random k-of-n patient draw misses this type entirely
        miss = 1.0
        for i in range(target_test_patients):
            miss *= max(0.0, (n_pat - k - i)) / max(1, (n_pat - i))
        flag = "  <-- RISK" if miss > 0.05 else ""
        print(f"  {str(t)[:40]:<40} in {k:>3}/{n_pat} patients | "
              f"P(absent from random test set) = {miss:6.1%}{flag}")
    print()

    # ---- how many patients hold ALL types ---------------------------------
    types = set(kept[T].unique())
    full = kept.groupby(P)[T].apply(lambda s: set(s) == types)
    print(f"  patients carrying ALL {len(types)} note types: "
          f"{int(full.sum())} of {n_pat}")
    print()

    # ---- per-patient summary table ----------------------------------------
    print("-- per-patient detail (sorted by note count) --")
    tab = kept.pivot_table(index=P, columns=T, values=args.id_col,
                           aggfunc="count", fill_value=0)
    tab["TOTAL"] = tab.sum(axis=1)
    if L in kept.columns:
        tab["LABEL"] = kept.groupby(P)[L].agg(lambda s: s.mode().iat[0])
    tab = tab.sort_values("TOTAL", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tab.to_string())
    print()

    print("=" * 70)
    print("READ THIS AS:")
    print("  * any type flagged RISK cannot be left to a random patient draw")
    print("    -- the splitter must place its carriers deliberately")
    print("  * if one patient holds >15% of notes, they alone can blow the")
    print("    70/15/15 note-count target -- they must go to train")
    print("  * if few patients carry all types, constraints are tight and")
    print("    exact 70/15/15 by notes may not be reachable")
    print("=" * 70)


if __name__ == "__main__":
    main()
