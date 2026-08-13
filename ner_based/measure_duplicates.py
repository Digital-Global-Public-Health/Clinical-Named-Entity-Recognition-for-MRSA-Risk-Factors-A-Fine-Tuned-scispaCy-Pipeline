#!/usr/bin/env python
"""
Measure exact- and near-duplicate rate in the AIR-MS note corpus.

Why: clinical notes copy forward heavily. Duplicates inflate the corpus without
adding signal, bias the student model toward repeated phrasings, and -- if
near-duplicates cross patients -- break the assumption that a patient-level
train/test split prevents leakage.

Method
  Tier 1  exact duplicates      : hash of the normalized full text
  Tier 2  near duplicates       : MinHash (128 perms) over 5-word shingles,
                                  banded LSH (32 bands x 4 rows) for candidate
                                  generation, union-find for clustering
  Tier 3  cross-patient clusters: near-duplicate clusters spanning >1 PERSON_ID

Only pandas + numpy required. Runs on a compute node in a few minutes for ~21k
notes. Reads text but writes only IDs -- no note text leaves the parquet.

Usage
  python measure_duplicates.py \
      --inputs /sc/arion/work/rademt02/airms_notes/extracted/cohort_all_notes.parquet

  # or the three batch draws
  python measure_duplicates.py --inputs .../batch01.parquet .../batch02.parquet .../batch03.parquet

  # write cluster assignments for later deduplication
  python measure_duplicates.py --inputs ... --out-csv dup_clusters.csv
"""

import argparse
import hashlib
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

WS = re.compile(r"\s+")
DIGITS = re.compile(r"\d")

# MinHash / LSH parameters. 32 bands x 4 rows targets ~0.6 Jaccard as the
# knee of the S-curve, so genuine copy-forward pairs surface as candidates.
NUM_PERM = 128
BANDS = 32
ROWS = NUM_PERM // BANDS
PRIME = 4294967311  # smallest prime > 2**32
SHINGLE_WORDS = 5


def normalize(text: str, mask_digits: bool) -> str:
    """Lowercase, collapse whitespace, optionally blank out numbers.

    Masking digits makes copy-forward text visible even when vitals, dates and
    timestamps have been refreshed. It also merges genuinely distinct notes that
    differ only in numbers, so report both settings if the rates diverge.
    """
    t = text.lower()
    if mask_digits:
        t = DIGITS.sub("0", t)
    return WS.sub(" ", t).strip()


def shingle_hashes(norm_text: str) -> np.ndarray:
    """32-bit hashes of overlapping 5-word shingles."""
    words = norm_text.split()
    if len(words) < SHINGLE_WORDS:
        grams = [" ".join(words)] if words else []
    else:
        grams = [
            " ".join(words[i : i + SHINGLE_WORDS])
            for i in range(len(words) - SHINGLE_WORDS + 1)
        ]
    seen = {
        int.from_bytes(hashlib.blake2b(g.encode("utf8"), digest_size=4).digest(), "big")
        for g in grams
    }
    return np.fromiter(seen, dtype=np.uint64, count=len(seen))


def signatures(hash_sets, a, b):
    """MinHash signature matrix, shape (n_docs, NUM_PERM)."""
    sig = np.full((len(hash_sets), NUM_PERM), np.iinfo(np.uint32).max, dtype=np.uint64)
    for i, h in enumerate(hash_sets):
        if h.size == 0:
            continue
        # a, b < 2**31 and h < 2**32 keeps the product inside uint64.
        perm = (a[:, None] * h[None, :] + b[:, None]) % PRIME
        sig[i] = perm.min(axis=1)
    return sig.astype(np.uint32)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="parquet file(s)")
    ap.add_argument("--text-col", default="NOTE_TEXT")
    ap.add_argument("--id-col", default="NOTE_ID")
    ap.add_argument("--patient-col", default="PERSON_ID")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="skip notes shorter than this (matches the batch draws)")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="estimated Jaccard above which two notes are near-duplicates")
    ap.add_argument("--mask-digits", action="store_true",
                    help="normalize all digits to 0 before hashing")
    ap.add_argument("--sample", type=int, default=0, help="use only N notes (quick run)")
    ap.add_argument("--out-csv", default=None, help="write note_id,person_id,cluster_id")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    frames = [pd.read_parquet(p, columns=[args.id_col, args.patient_col, args.text_col])
              for p in args.inputs]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=[args.id_col])

    total_raw = len(df)
    df = df[df[args.text_col].str.len() >= args.min_chars].reset_index(drop=True)
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=args.seed).reset_index(drop=True)

    n = len(df)
    print(f"notes read              : {total_raw}")
    print(f"notes >= {args.min_chars} chars      : {n}")
    print(f"patients                : {df[args.patient_col].nunique()}")
    print(f"digit masking           : {args.mask_digits}")
    print()

    norm = [normalize(t, args.mask_digits) for t in df[args.text_col]]

    # ---- Tier 1: exact duplicates ------------------------------------------
    exact = Counter(hashlib.blake2b(t.encode("utf8"), digest_size=16).hexdigest()
                    for t in norm)
    exact_dupes = sum(c - 1 for c in exact.values() if c > 1)
    print("== exact duplicates (identical after normalization) ==")
    print(f"distinct texts          : {len(exact)}")
    print(f"redundant copies        : {exact_dupes}  ({100*exact_dupes/n:.1f}% of notes)")
    print()

    # ---- Tier 2: near duplicates -------------------------------------------
    rng = np.random.default_rng(args.seed)
    a = rng.integers(1, 2**31, size=NUM_PERM, dtype=np.uint64)
    b = rng.integers(0, 2**31, size=NUM_PERM, dtype=np.uint64)

    print("hashing shingles ...", flush=True)
    hash_sets = [shingle_hashes(t) for t in norm]
    print("computing MinHash signatures ...", flush=True)
    sig = signatures(hash_sets, a, b)

    print("banding + clustering ...", flush=True)
    uf = UnionFind(n)
    comparisons = 0
    for band in range(BANDS):
        cols = sig[:, band * ROWS : (band + 1) * ROWS]
        buckets = defaultdict(list)
        for i, row in enumerate(cols):
            buckets[row.tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            # Compare each member against the bucket representative rather than
            # all pairs: linear instead of quadratic, adequate for a rate.
            rep = members[0]
            for other in members[1:]:
                if uf.find(rep) == uf.find(other):
                    continue
                est = float(np.mean(sig[rep] == sig[other]))
                comparisons += 1
                if est >= args.threshold:
                    uf.union(rep, other)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    redundant = n - len(clusters)

    print()
    print(f"== near duplicates (estimated Jaccard >= {args.threshold}) ==")
    print(f"signature comparisons   : {comparisons}")
    print(f"clusters (unique content): {len(clusters)}")
    print(f"redundant notes         : {redundant}  ({100*redundant/n:.1f}% of notes)")
    print(f"effective corpus size   : {len(clusters)} of {n}")
    print(f"largest cluster sizes   : {sizes[:15]}")
    print(f"clusters with >1 note   : {sum(1 for s in sizes if s > 1)}")
    print()

    # ---- Tier 3: cross-patient clusters ------------------------------------
    pids = df[args.patient_col].to_numpy()
    cross = [(root, len(idx), len(set(pids[idx])))
             for root, idx in clusters.items()
             if len(set(pids[idx])) > 1]
    cross_notes = sum(c[1] for c in cross)
    print("== cross-patient near-duplicate clusters ==")
    print(f"clusters spanning >1 patient : {len(cross)}")
    print(f"notes involved               : {cross_notes}  ({100*cross_notes/n:.1f}% of notes)")
    if cross:
        top = sorted(cross, key=lambda c: -c[1])[:10]
        print("  largest (notes, patients):", [(c[1], c[2]) for c in top])
        print("  NOTE: a patient-level split does NOT remove this leakage.")
    else:
        print("  none -- a patient-level split is sufficient to prevent leakage.")

    if args.out_csv:
        cluster_id = {}
        for cid, (root, idx) in enumerate(clusters.items()):
            for i in idx:
                cluster_id[i] = cid
        out = pd.DataFrame({
            "note_id": df[args.id_col],
            "person_id": df[args.patient_col],
            "cluster_id": [cluster_id[i] for i in range(n)],
        })
        out["is_cluster_representative"] = ~out.duplicated(subset=["cluster_id"])
        out.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv} "
              f"({int(out.is_cluster_representative.sum())} representatives)")


if __name__ == "__main__":
    main()
