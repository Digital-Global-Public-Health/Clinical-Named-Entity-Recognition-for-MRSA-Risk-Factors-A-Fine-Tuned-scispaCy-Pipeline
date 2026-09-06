#!/usr/bin/env python
"""Rewrite WebAnno TSV3 exports from the built-in NamedEntity layer onto a
custom span layer that carries the five assertion features.

Why: INCEpTION does not allow adding features to built-in layers (documented
restriction; see inception-project/inception#1798). The gold spans in project 4
live on de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity, which has only
`identifier` and `value`. To annotate polarity / certainty / temporality /
experiencer / allergy we move the same spans onto a custom layer.

What it does, per file:
  - rewrites the #T_SP= header line for the NamedEntity layer to the custom
    type, replacing |identifier|value with |value|<five features>
  - for every token row, replaces that layer's two columns with six: the
    original `value`, then five nulls (`*`, carrying the same [n]
    disambiguation suffix so multi-token spans stay grouped)

Everything else — token ids, offsets, text, sentence markers, other layers —
is passed through untouched.

RUN ON A MINERVA COMPUTE NODE. The TSVs contain note text (PHI).

Usage:
    python rewrite_tsv_layer.py --in-dir gold_tsv --out-dir gold_tsv_assert
    python rewrite_tsv_layer.py --in-dir gold_tsv --out-dir out --only 110577574
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

OLD_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"
NEW_TYPE = "webanno.custom.Assertion"
NEW_FEATURES = ["value", "polarity", "certainty", "temporality",
                "experiencer", "allergy"]
# Written into every imported span so the annotator only touches exceptions.
# INCEpTION's per-feature "Default value" applies to spans created in the UI,
# not to spans arriving via import -- hence baking them in here.
DEFAULTS = ["affirmed", "certain", "current", "patient", "no"]

LAYER_PREFIXES = ("#T_SP=", "#T_CH=", "#T_RL=")


def parse_header(lines):
    """Return (layer_decls, header_line_indices).

    layer_decls is a list of (prefix, type_name, [features]) in file order --
    which is also the column order after the three fixed columns.
    """
    decls, idxs = [], []
    for i, ln in enumerate(lines):
        for pfx in LAYER_PREFIXES:
            if ln.startswith(pfx):
                body = ln[len(pfx):].rstrip("\n")
                parts = body.split("|")
                decls.append((pfx, parts[0], parts[1:]))
                idxs.append(i)
                break
    return decls, idxs


def default_cells(value_cell: str) -> list:
    """The five attribute cells for one token, carrying any [n] suffix."""
    if value_cell == "_":
        return ["_"] * len(DEFAULTS)
    suffix = ""
    if value_cell.endswith("]") and "[" in value_cell:
        suffix = value_cell[value_cell.rindex("["):]
    return [d + suffix for d in DEFAULTS]


def rewrite(text: str) -> tuple[str, int]:
    lines = text.split("\n")
    decls, idxs = parse_header(lines)

    target = None
    col = 3  # id, offsets, token
    for k, (pfx, tname, feats) in enumerate(decls):
        if tname == OLD_TYPE:
            target = (k, col, len(feats), feats)
            break
        col += len(feats)
    if target is None:
        raise SystemExit(f"no {OLD_TYPE} layer found in header")

    k, start, n_old, old_feats = target
    if old_feats != ["identifier", "value"]:
        print(f"  ! unexpected feature list {old_feats}", file=sys.stderr)
    value_off = old_feats.index("value")

    lines[idxs[k]] = "#T_SP=" + NEW_TYPE + "|" + "|".join(NEW_FEATURES)

    changed = 0
    for i, ln in enumerate(lines):
        if not ln or ln.startswith("#"):
            continue
        cols = ln.split("\t")
        trailing = cols and cols[-1] == ""
        if trailing:
            cols = cols[:-1]
        if len(cols) < start + n_old:
            continue
        block = cols[start:start + n_old]
        value = block[value_off]
        cols[start:start + n_old] = [value] + default_cells(value)
        lines[i] = "\t".join(cols) + ("\t" if trailing else "")
        changed += 1

    return "\n".join(lines), changed


LABEL_RE = re.compile(r"^(DISEASE|MEDICATION|PROCEDURE)(\[[0-9]+\])?$")


def has_spans(text: str) -> bool:
    """True if any token row carries an entity label (skips opened-but-empty docs)."""
    for ln in text.split("\n"):
        if not ln or ln.startswith("#"):
            continue
        if any(LABEL_RE.match(c) for c in ln.split("\t")[3:]):
            return True
    return False


def note_id_of(path) -> str:
    """annotation/110577574.txt/admin.tsv -> 110577574"""
    return path.parent.name.replace(".txt", "") or path.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default=None, help="process only this note id")
    args = ap.parse_args()

    src = pathlib.Path(args.in_dir)
    dst = pathlib.Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Export layout: annotation/<note_id>.txt/admin.tsv
    files = sorted(src.rglob("admin.tsv"))
    if not files:
        files = sorted(src.glob("*.tsv"))
    if args.only:
        files = [f for f in files if args.only in str(f)]
    if not files:
        sys.exit(f"no annotation files matched in {src}")

    written = 0
    for f in files:
        text = f.read_text()
        if not has_spans(text):
            print(f"{note_id_of(f)}: no spans, skipped")
            continue
        out, n = rewrite(text)
        (dst / (note_id_of(f) + ".tsv")).write_text(out)
        print(f"{note_id_of(f)}.tsv: {n} token rows rewritten")
        written += 1
    print(f"\n{written} files written to {dst}")


if __name__ == "__main__":
    main()
