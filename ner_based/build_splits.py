#!/usr/bin/env python
"""
Build patient-level train/dev/test splits from the AIR-MS silver corpus.

Rebuilds DocBins from the verified pre-annotation JSONs (single source of
truth) rather than merging the per-batch DocBins, so filtering decisions are
explicit and identical across all splits.

Priority of requirements (deliberate, documented order):
  1. HARD  patient integrity -- a patient's notes never straddle splits
  2. OPT   note-count proportions (default 70/15/15), greedy longest-first
  3. TIE   label (case/control) balance, used only to break ties
  4. CHECK all note types present in every split -- asserted, not optimised
           (48/50 patients carry all 4 types, so this is free)
  5. FIXED seed + deterministic assignment; no searching for a lucky partition

Outputs (all under --out-dir):
  manifest.csv          canonical: one row per note, join key for everything
  train.spacy dev.spacy test.spacy   DocBins, note_id stored in user_data
  train_{2000,5000,10000}.spacy      nested subsets for the learning curve
  gold_candidates.csv   test-split notes, type-stratified, for hand annotation
  split_report.md       counts per split x type x label, entity counts, seed

Governance: manifest.csv and gold_candidates.csv contain PERSON_ID / NOTE_ID.
Keep them inside the enclave and gitignore them; split_report.md is aggregate
counts only and is safe to commit.

Usage
  python build_splits.py \
      --ann-dirs annotations/batch01_v2 annotations/batch02 annotations/batch03 \
      --parquet /sc/arion/work/rademt02/airms_notes/extracted/cohort_all_notes.parquet \
      --out-dir splits
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

TEXT_KEYS = ("text", "note_text", "source_text", "NOTE_TEXT")
SPAN_KEYS = ("spans", "entities", "verified_spans", "verified")


def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None


def load_annotations(ann_dirs):
    """Read every verified JSON. note_id comes from the filename stem."""
    recs = {}
    for ad in ann_dirs:
        files = sorted(Path(ad).glob("verified/*.json"))
        if not files:
            files = sorted(Path(ad).glob("*.json"))
        for f in files:
            d = json.load(open(f))
            text = pick(d, TEXT_KEYS) or ""
            spans = pick(d, SPAN_KEYS) or []
            nid = str(d.get("note_id") or f.stem)
            recs[nid] = {"note_id": nid, "text": text, "spans": spans,
                         "batch": Path(ad).name}
        print(f"  {ad}: {len(files)} files")
    return recs


def assign_patients(pat_df, fracs, seed):
    """Greedy longest-processing-time assignment of whole patients.

    Largest patients are placed first, when there is still capacity to absorb
    them; small patients then fill the gaps. Ties on remaining note capacity
    are broken toward whichever split is furthest below its label target.
    """
    total = pat_df.n_notes.sum()
    targets = {s: total * f for s, f in fracs.items()}
    assigned = {s: 0 for s in fracs}
    lab_assigned = {s: Counter() for s in fracs}
    lab_targets = {
        s: {l: pat_df[pat_df.label == l].n_notes.sum() * f for l in pat_df.label.unique()}
        for s, f in fracs.items()
    }
    out = {}

    ordered = pat_df.sort_values(
        ["n_notes", "person_id"], ascending=[False, True]
    ).itertuples()

    for p in ordered:
        best, best_key = None, None
        for s in fracs:
            room = targets[s] - assigned[s]
            lab_room = lab_targets[s].get(p.label, 0) - lab_assigned[s][p.label]
            key = (round(room, 6), round(lab_room, 6))
            if best_key is None or key > best_key:
                best, best_key = s, key
        out[p.person_id] = best
        assigned[best] += p.n_notes
        lab_assigned[best][p.label] += p.n_notes

    return out, assigned, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dirs", nargs="+", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--base-model", default="en_core_sci_sm",
                    help="tokenizer source; falls back to blank 'en' if absent")
    ap.add_argument("--fracs", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "DEV", "TEST"))
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--curve-sizes", nargs="*", type=int, default=[2000, 5000, 10000])
    ap.add_argument("--gold-quota", nargs="*", default=[
        "Progress Notes=10", "Consults=6", "H&P=5", "Discharge Summary=4"])
    ap.add_argument("--low-power-notes", type=int, default=50)
    ap.add_argument("--dup-csv", default=None,
                    help="optional dup_clusters.csv to carry cluster ids into the manifest")
    ap.add_argument("--drop-cross-patient-dupes", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import spacy
    from spacy.tokens import DocBin
    from spacy.util import filter_spans

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # ---------------------------------------------------------------- load
    print("loading annotations ...")
    recs = load_annotations(args.ann_dirs)
    print(f"  total annotated notes: {len(recs)}")

    meta = pd.read_parquet(args.parquet)
    meta["note_id"] = meta.NOTE_ID.astype(str)
    meta = meta[["note_id", "PERSON_ID", "NOTE_TITLE", "LABEL", "NOTE_TEXT"]].copy()
    meta["n_chars"] = meta.NOTE_TEXT.str.len()
    meta = meta.drop(columns=["NOTE_TEXT"]).rename(
        columns={"PERSON_ID": "person_id", "NOTE_TITLE": "note_type", "LABEL": "label"})

    df = meta[meta.note_id.isin(recs)].copy()
    df = df[df.n_chars >= args.min_chars].reset_index(drop=True)
    print(f"  joined + >= {args.min_chars} chars: {len(df)} notes, "
          f"{df.person_id.nunique()} patients")

    missing = set(recs) - set(meta.note_id)
    if missing:
        print(f"  WARNING: {len(missing)} annotated notes not found in parquet")

    if args.dup_csv:
        dup = pd.read_csv(args.dup_csv, dtype={"note_id": str})
        df = df.merge(dup[["note_id", "cluster_id", "is_cluster_representative"]],
                      on="note_id", how="left")
        if args.drop_cross_patient_dupes:
            cross = (dup.groupby("cluster_id").person_id.nunique() > 1)
            bad = set(dup[dup.cluster_id.isin(cross[cross].index)].note_id)
            before = len(df)
            df = df[~df.note_id.isin(bad)]
            print(f"  dropped {before - len(df)} cross-patient duplicate notes")

    # ------------------------------------------------------------- assign
    pat = (df.groupby("person_id")
             .agg(n_notes=("note_id", "size"),
                  label=("label", lambda s: s.mode().iat[0]))
             .reset_index())
    fracs = {"train": args.fracs[0], "dev": args.fracs[1], "test": args.fracs[2]}
    mapping, assigned, targets = assign_patients(pat, fracs, args.seed)
    df["split"] = df.person_id.map(mapping)

    print("\nsplit assignment (by notes):")
    for s in fracs:
        n = int((df.split == s).sum())
        npat = df[df.split == s].person_id.nunique()
        print(f"  {s:<6} {n:>6} notes ({100*n/len(df):5.1f}%, target "
              f"{100*fracs[s]:.0f}%)  {npat:>3} patients")

    # --------------------------------------------- requirement 4: assert
    all_types = set(df.note_type.unique())
    problems = []
    for s in fracs:
        present = set(df[df.split == s].note_type.unique())
        if present != all_types:
            problems.append(f"{s} missing {sorted(all_types - present)}")
    if problems:
        raise SystemExit("NOTE TYPE CHECK FAILED: " + "; ".join(problems))
    print(f"\n[OK] all {len(all_types)} note types present in every split")

    # ------------------------------------------------------- build docs
    try:
        nlp = spacy.load(args.base_model, disable=["ner", "parser", "tagger"])
        tok_src = args.base_model
    except Exception:
        nlp = spacy.blank("en")
        tok_src = "blank:en (WARNING: retokenize if training on scispaCy)"
    print(f"tokenizer: {tok_src}")

    stats = Counter()
    docs_by_split = defaultdict(list)
    ents_per_note = {}

    for r in df.itertuples():
        rec = recs[r.note_id]
        doc = nlp.make_doc(rec["text"])
        spans = []
        for s in rec["spans"]:
            lab = s.get("label")
            if lab not in {"DISEASE", "MEDICATION", "PROCEDURE"}:
                stats["skipped_invalid_label"] += 1
                continue
            sp = doc.char_span(s["start_char"], s["end_char"], label=lab,
                               alignment_mode="expand")
            if sp is None:
                stats["alignment_failed"] += 1
                continue
            spans.append(sp)
        kept = filter_spans(spans)
        stats["dropped_overlapping"] += len(spans) - len(kept)
        doc.ents = kept
        doc.user_data["note_id"] = r.note_id
        doc.user_data["person_id"] = str(r.person_id)
        doc.user_data["note_type"] = r.note_type
        docs_by_split[r.split].append(doc)
        ents_per_note[r.note_id] = len(kept)
        stats["entities"] += len(kept)

    df["n_entities"] = df.note_id.map(ents_per_note).fillna(0).astype(int)

    for s in fracs:
        db = DocBin(store_user_data=True, docs=docs_by_split[s])
        db.to_disk(out / f"{s}.spacy")
        print(f"  wrote {s}.spacy: {len(docs_by_split[s])} docs, "
              f"{sum(len(d.ents) for d in docs_by_split[s])} entities")

    # --------------------------------------- nested learning-curve subsets
    train_docs = docs_by_split["train"][:]
    random.Random(args.seed).shuffle(train_docs)
    for n in args.curve_sizes:
        if n >= len(train_docs):
            continue
        DocBin(store_user_data=True, docs=train_docs[:n]).to_disk(out / f"train_{n}.spacy")
        print(f"  wrote train_{n}.spacy ({n} docs, nested)")

    # -------------------------------------------------- gold candidates
    quota = {}
    for q in args.gold_quota:
        k, v = q.rsplit("=", 1)
        quota[k] = int(v)
    test = df[df.split == "test"].copy()
    picks = []
    for t, k in quota.items():
        sub = test[test.note_type == t]
        if sub.empty:
            continue
        # spread across patients: one per patient before a second from any
        sub = sub.sample(frac=1, random_state=args.seed)
        sub["rank_in_patient"] = sub.groupby("person_id").cumcount()
        sub = sub.sort_values(["rank_in_patient", "n_entities"],
                              ascending=[True, False])
        picks.append(sub.head(k))
    gold = pd.concat(picks) if picks else test.head(0)
    gold[["note_id", "person_id", "note_type", "label", "n_chars", "n_entities"]] \
        .to_csv(out / "gold_candidates.csv", index=False)
    print(f"  wrote gold_candidates.csv: {len(gold)} notes, "
          f"{gold.person_id.nunique()} patients, {gold.note_type.nunique()} types")

    # ------------------------------------------------------- manifest
    cols = ["note_id", "person_id", "note_type", "label", "n_chars",
            "n_entities", "split"]
    if "cluster_id" in df.columns:
        cols += ["cluster_id", "is_cluster_representative"]
    df[cols].to_csv(out / "manifest.csv", index=False)

    # --------------------------------------------------------- report
    lines = ["# Split report", "",
             f"- seed: {args.seed}",
             f"- targets: {fracs}",
             f"- tokenizer: {tok_src}",
             f"- min_chars: {args.min_chars}",
             f"- notes: {len(df)} | patients: {df.person_id.nunique()} | "
             f"entities: {stats['entities']}", ""]

    lines += ["## Notes and entities per split", "",
              "| split | patients | notes | % | entities |", "|---|---|---|---|---|"]
    for s in fracs:
        d = df[df.split == s]
        lines.append(f"| {s} | {d.person_id.nunique()} | {len(d)} | "
                     f"{100*len(d)/len(df):.1f}% | {int(d.n_entities.sum())} |")

    lines += ["", "## Note type x split (notes / entities)", "",
              "| type | " + " | ".join(fracs) + " |",
              "|---|" + "---|" * len(fracs)]
    for t in sorted(all_types):
        row = [t]
        for s in fracs:
            d = df[(df.split == s) & (df.note_type == t)]
            flag = " (LOW POWER)" if s == "test" and len(d) < args.low_power_notes else ""
            row.append(f"{len(d)} / {int(d.n_entities.sum())}{flag}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Label balance", "",
              "| split | " + " | ".join(str(l) for l in sorted(df.label.unique())) + " |",
              "|---|" + "---|" * df.label.nunique()]
    for s in fracs:
        d = df[df.split == s]
        lines.append("| " + s + " | " +
                     " | ".join(str(int((d.label == l).sum()))
                               for l in sorted(df.label.unique())) + " |")

    lines += ["", "## Build statistics", ""]
    for k, v in sorted(stats.items()):
        lines.append(f"- {k}: {v}")
    lines += ["",
              "Note: `alignment_mode='expand'` snaps model character offsets to "
              "token boundaries rather than discarding the span; "
              "`dropped_overlapping` are spans spaCy cannot store in `doc.ents`.",
              ""]
    (out / "split_report.md").write_text("\n".join(lines))

    print(f"\nwrote {out}/manifest.csv and {out}/split_report.md")
    print("Reminder: gitignore manifest.csv and gold_candidates.csv (contain IDs).")


if __name__ == "__main__":
    main()
