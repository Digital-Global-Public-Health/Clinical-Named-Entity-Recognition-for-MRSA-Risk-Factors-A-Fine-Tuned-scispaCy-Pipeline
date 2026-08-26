#!/usr/bin/env python
"""Apply lexicon.yaml to the corpus vocabulary and report what it catches.

This is the review loop for lexicon curation: run it, read the matched terms
for one feature, add missing surface forms to `include` or false positives to
`exclude`, run it again. The corpus decides the vocabulary, not memory.

Three reports:
  --report coverage   per-feature term and occurrence counts (default)
  --report terms      the matched terms for one or more features, ranked
  --report unmapped   frequent terms no feature claims -- the gap list

RUN ON A MINERVA COMPUTE NODE (span text is PHI-adjacent).

Usage:
    python review_lexicon.py --candidates lexicon_candidates.csv
    python review_lexicon.py --report terms --feature dialysis_access
    python review_lexicon.py --report unmapped --top 60
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

VALID_LABELS = {"DISEASE", "MEDICATION", "PROCEDURE"}
REF = re.compile(r"^__ref__:(.+)$")


def load_features(path):
    import yaml
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    feats = doc["features"]
    by_name = {f["name"]: f for f in feats}

    def expand(name, seen):
        if name in seen:
            sys.exit(f"circular __ref__ at {name}")
        seen = seen | {name}
        out = []
        for pat in by_name[name].get("include") or []:
            m = REF.match(pat)
            if m:
                if m.group(1) not in by_name:
                    sys.exit(f"{name}: __ref__ to unknown feature {m.group(1)}")
                out.extend(expand(m.group(1), seen))
            else:
                out.append(pat)
        return out

    for f in feats:
        pats = expand(f["name"], set())
        f["_inc"] = [re.compile(p, re.I) for p in pats]
        f["_exc"] = [re.compile(p, re.I) for p in (f.get("exclude") or [])]
        f["_types"] = set(f.get("entity_types") or VALID_LABELS)
    return feats


def load_candidates(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["label"] not in VALID_LABELS:
                continue
            rows.append((r["label"], r["text"], int(r["n"])))
    return rows


def match(feat, label, text):
    if label not in feat["_types"]:
        return False
    if not any(p.search(text) for p in feat["_inc"]):
        return False
    if any(p.search(text) for p in feat["_exc"]):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="lexicon.yaml")
    ap.add_argument("--candidates", default="lexicon_candidates.csv")
    ap.add_argument("--report", default="coverage",
                    choices=["coverage", "terms", "unmapped"])
    ap.add_argument("--feature", action="append", default=None)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    feats = load_features(args.lexicon)
    rows = load_candidates(args.candidates)
    total_occ = sum(n for _, _, n in rows)

    hits = {f["name"]: [] for f in feats}
    claimed = set()
    for label, text, n in rows:
        for f in feats:
            if match(f, label, text):
                hits[f["name"]].append((n, label, text))
                claimed.add((label, text))

    if args.report == "coverage":
        print(f"{len(rows)} distinct terms, {total_occ} occurrences\n")
        print(f"{'feature':<30} {'type':<12} {'terms':>6} {'occ':>8} {'%':>6}")
        for f in feats:
            h = hits[f["name"]]
            occ = sum(n for n, _, _ in h)
            t = "/".join(sorted(x[:4] for x in f["_types"]))
            warn = "  <- EMPTY" if not h else ""
            print(f"{f['name']:<30} {t:<12} {len(h):>6} {occ:>8} "
                  f"{100*occ/total_occ:>5.1f}%{warn}")
        cl_occ = sum(n for label, text, n in rows if (label, text) in claimed)
        print(f"\nclaimed by >=1 feature: {len(claimed)} terms, "
              f"{cl_occ} occurrences ({100*cl_occ/total_occ:.1f}%)")

    elif args.report == "terms":
        names = args.feature or [f["name"] for f in feats]
        for name in names:
            if name not in hits:
                print(f"! unknown feature {name}", file=sys.stderr)
                continue
            h = sorted(hits[name], reverse=True)
            print(f"\n=== {name}: {len(h)} terms, "
                  f"{sum(n for n, _, _ in h)} occurrences ===")
            for n, label, text in h[:args.top]:
                print(f"  {n:>6}  {label:<11} {text}")
            if len(h) > args.top:
                print(f"  ... {len(h)-args.top} more")

    else:  # unmapped
        un = [(n, label, text) for label, text, n in rows
              if (label, text) not in claimed]
        un.sort(reverse=True)
        occ = sum(n for n, _, _ in un)
        print(f"{len(un)} unmapped terms, {occ} occurrences "
              f"({100*occ/total_occ:.1f}% of corpus)\n")
        for n, label, text in un[:args.top]:
            print(f"  {n:>6}  {label:<11} {text}")


if __name__ == "__main__":
    main()
