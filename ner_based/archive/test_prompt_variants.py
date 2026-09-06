#!/usr/bin/env python
"""
Prompt / multi-pass experiment against the 6-note gold set.

Answers one question cheaply, before committing GPU time to a 21k-note re-run:
does the teacher's 35 % recall improve more from (a) reframing the prompt as an
exhaustive extraction task, or (b) running the current prompt several times and
unioning the results?

Runs entirely inside Minerva (reads real note text -> local Ollama). Compares
every variant against `gold.spacy` using the same span-matching rules as
teacher_vs_gold.py, so the numbers sit directly beside the existing baseline
(P 75.94 / R 35.25 / F 48.15).

Setup
  # login node: launch Ollama, note the Access URL + token
  export OLLAMA_HOST=http://10.95.46.93:<port>
  export OLLAMA_AUTH_USER=rademt02
  export OLLAMA_AUTH_TOKEN=$(cut -d: -f2 ~/.ollama_secure/authorized_users.txt)

Usage
  python test_prompt_variants.py --gold annotations/gold_export/gold.spacy \
      --variants current exhaustive --passes 1
  python test_prompt_variants.py --gold annotations/gold_export/gold.spacy \
      --variants current --passes 3          # multi-pass union
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter

# --------------------------------------------------------------------------
# Prompt A: the current production prompt, verbatim from src/ner/preannotate.py
# --------------------------------------------------------------------------
CURRENT = """You propose clinical named entity mentions for human review.

Labels:
- DISEASE: diagnoses, clinical conditions, infections, symptoms, and chronic illnesses. Examples: MRSA, pneumonia, cellulitis, sepsis, diabetes mellitus, CHF, RA, COPD, fever.
- MEDICATION: drugs, drug classes, antibiotics, immunosuppressants, and allergens when the allergen is a medication. Examples: vancomycin, prednisone, methotrexate, penicillin, antibiotics.
- PROCEDURE: clinical procedures, surgical interventions, invasive devices, indwelling devices, and lines. Indwelling devices and lines ARE procedures. Examples: central line, PICC, tunneled cath, Foley, urinary catheter, hemodialysis/HD, intubation, tracheostomy, CABG, wound debridement.

Return JSON only, with this shape:
{
  "entities": [
    {
      "text": "verbatim text copied exactly from the note",
      "label": "DISEASE|MEDICATION|PROCEDURE"
    }
  ]
}

Rules:
- The "text" value must be copied verbatim from the note. Do not paraphrase.
- Include only these two keys per entity: text and label. Do not add offsets or attributes.
- Do NOT omit negated, uncertain, historical, or family-member mentions. Tag the entity regardless of context.
- ALLERGY: "Penicillin allergy" -> tag "Penicillin" as MEDICATION.
- Do NOT tag anatomy or body parts (e.g. "L leg", "left arm"). Not entities.

Span boundaries (tag the shortest span that carries the clinical meaning):
- MEDICATION: drug name only. Exclude parenthetical brand names, dose, strength,
  form, route, and frequency.
  "omeprazole (PRILOSEC) 20 mg capsule Take 20 mg by mouth daily" -> "omeprazole"
- DISEASE/PROCEDURE: tag the full clinical term, but exclude trailing anatomical
  qualifiers that are separate body-part mentions.
  "Macular degeneration of right eye" -> "Macular degeneration"
- Where an abbreviation and its expansion both appear, tag each separately.
- Exclude leading bullets, numbering, and section labels from the span.

Few-shot example:
Note: No pneumonia today. Mother had lymphoma. Penicillin allergy listed. If fever develops, start vancomycin.
JSON:
{
  "entities": [
    {"text": "pneumonia", "label": "DISEASE"},
    {"text": "lymphoma", "label": "DISEASE"},
    {"text": "Penicillin", "label": "MEDICATION"},
    {"text": "fever", "label": "DISEASE"},
    {"text": "vancomycin", "label": "MEDICATION"}
  ]
}
"""

# --------------------------------------------------------------------------
# Prompt B: reframed as EXHAUSTIVE extraction.
# Changes, each targeting an observed failure:
#   1. "extract every mention" replaces "propose ... for review"
#   2. explicit anti-summarisation instruction + realistic entity count
#   3. explicit "tag repeat mentions separately"
#   4. symptom examples expanded (the misses were overwhelmingly symptoms:
#      pain x11, SOB x4, back pain, chest pain, rashes)
#   5. few-shot replaced with a longer, denser example so the shot does not
#      anchor the model to a short output list
# --------------------------------------------------------------------------
EXHAUSTIVE = """You perform EXHAUSTIVE clinical named entity extraction. Your task is to find EVERY mention of every clinical entity in the note - not a summary, not a selection, not the important ones. All of them.

Labels:
- DISEASE: diagnoses, clinical conditions, infections, syndromes, chronic illnesses, AND symptoms/findings. Examples: MRSA, pneumonia, cellulitis, sepsis, diabetes mellitus, CHF, RA, COPD, fever, pain, back pain, chest pain, abdominal pain, SOB, shortness of breath, hypoxia, nausea, rash, rashes, swelling, weakness, fatigue, dizziness, hyponatremia, thrombus, PE, PTX, abscess, lesions, obstruction.
- MEDICATION: drugs, drug classes, antibiotics, immunosuppressants, and allergens when the allergen is a medication. Examples: vancomycin, daptomycin, Zosyn, Bactrim, Tylenol, prednisone, methotrexate, penicillin, antibiotics, opioids.
- PROCEDURE: clinical procedures, surgical interventions, imaging studies, invasive devices, indwelling devices, and lines. Devices and lines ARE procedures. Examples: central line, PICC, tunneled cath, Foley, urinary catheter, hemodialysis/HD, intubation, tracheostomy, chest tube, CABG, wound debridement, MRI, CT, CTA, TTE, TEE, X-ray, biopsy.

CRITICAL - completeness:
- Do NOT summarise. Do NOT select representative entities. A long clinical note routinely contains 100-300 entity mentions; extract all of them.
- Scan the ENTIRE note top to bottom: history, HPI, physical exam, medication lists, imaging reports, assessment and plan. Entities in medication lists and exam findings count exactly as much as those in the assessment.
- If the same entity is mentioned 5 times, output it 5 times - one object per mention.
- An empty or short output list is almost always WRONG for a note longer than a few sentences.

Return JSON only, with this shape:
{
  "entities": [
    {"text": "verbatim text copied exactly from the note", "label": "DISEASE|MEDICATION|PROCEDURE"}
  ]
}

Rules:
- The "text" value must be copied verbatim from the note, character for character, including any misspellings. Do not paraphrase or correct.
- Include only these two keys per entity: text and label.
- Do NOT omit negated, uncertain, historical, or family-member mentions. Tag the entity regardless of context. "no pneumothorax" -> tag "pneumothorax". "no LAD" -> tag "LAD".
- ALLERGY: "Penicillin allergy" -> tag "Penicillin" as MEDICATION.
- Do NOT tag anatomy or body parts alone (e.g. "L leg", "left arm"), lab test names or values (e.g. "WBC 12.3", "INR"), or normal findings with nothing named (e.g. "unremarkable", "MMM").

Span boundaries (tag the shortest span that carries the clinical meaning):
- MEDICATION: drug name only. Exclude parenthetical brand names, dose, strength, form, route, frequency.
  "omeprazole (PRILOSEC) 20 mg capsule Take 20 mg by mouth daily" -> "omeprazole"
  A brand name standing alone with no generic IS the drug name: "started on Bactrim" -> "Bactrim"
- DISEASE/PROCEDURE: tag the full clinical term, but exclude trailing anatomical qualifiers that are separate body-part mentions.
  "Macular degeneration of right eye" -> "Macular degeneration"
- Where an abbreviation and its expansion both appear, tag each separately.
- Exclude leading bullets, numbering, and section labels from the span.

Few-shot example:
Note:
HPI: 61F with PMHx of COPD, CHF, and diabetes presents with fever and SOB x 2 days. No chest pain. Blood cultures positive for MRSA.
Exam: no LAD, no JVD. Skin: no ulceration or rashes.
Imaging: CTA negative for PE, shows pleural effusion.
Plan: continue vancomycin and Zosyn. Foley in place. Pain controlled with morphine.
JSON:
{
  "entities": [
    {"text": "COPD", "label": "DISEASE"},
    {"text": "CHF", "label": "DISEASE"},
    {"text": "diabetes", "label": "DISEASE"},
    {"text": "fever", "label": "DISEASE"},
    {"text": "SOB", "label": "DISEASE"},
    {"text": "chest pain", "label": "DISEASE"},
    {"text": "MRSA", "label": "DISEASE"},
    {"text": "LAD", "label": "DISEASE"},
    {"text": "JVD", "label": "DISEASE"},
    {"text": "ulceration", "label": "DISEASE"},
    {"text": "rashes", "label": "DISEASE"},
    {"text": "CTA", "label": "PROCEDURE"},
    {"text": "PE", "label": "DISEASE"},
    {"text": "pleural effusion", "label": "DISEASE"},
    {"text": "vancomycin", "label": "MEDICATION"},
    {"text": "Zosyn", "label": "MEDICATION"},
    {"text": "Foley", "label": "PROCEDURE"},
    {"text": "Pain", "label": "DISEASE"},
    {"text": "morphine", "label": "MEDICATION"}
  ]
}
"""

PROMPTS = {"current": CURRENT, "exhaustive": EXHAUSTIVE}
VALID = {"DISEASE", "MEDICATION", "PROCEDURE"}


# ---------------------------------------------------------------- Ollama ---
def call_ollama(system_prompt, note_text, model, num_ctx, temperature, timeout=600):
    host = os.environ["OLLAMA_HOST"].rstrip("/")
    user = os.environ.get("OLLAMA_AUTH_USER", "")
    token = os.environ.get("OLLAMA_AUTH_TOKEN", "")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": note_text},
        ],
        "stream": False,
        "format": "json",
        "options": {"num_ctx": num_ctx, "temperature": temperature},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {user}:{token}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body.get("message", {}).get("content", "")


def parse_entities(raw):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    ents = obj.get("entities") if isinstance(obj, dict) else obj
    if not isinstance(ents, list):
        return None
    out = []
    for e in ents:
        if isinstance(e, dict) and e.get("label") in VALID and e.get("text"):
            out.append((str(e["text"]), e["label"]))
    return out


# ------------------------------------------------- span location (as prod) --
def find_occurrences(text, needle, ignore_case=False):
    if not needle:
        return []
    pattern = re.escape(needle)
    if needle[0].isalnum() or needle[0] == "_":
        pattern = r"(?<!\w)" + pattern
    if needle[-1].isalnum() or needle[-1] == "_":
        pattern = pattern + r"(?!\w)"
    flags = re.IGNORECASE if ignore_case else 0
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags)]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--variants", nargs="+", default=["current", "exhaustive"],
                    choices=list(PROMPTS))
    ap.add_argument("--passes", type=int, default=1,
                    help="runs per note; >1 combines results across passes")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="entity must appear in >= N passes (1 = union)")
    ap.add_argument("--model", default="llama3.3:70b")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="use ~0.7 with --passes>1 or every pass is identical")
    ap.add_argument("--ignore-case", action="store_true",
                    help="case-insensitive span location (recovers 'hyponatremia')")
    ap.add_argument("--base-model", default="en_core_sci_sm")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import spacy
    from spacy.tokens import DocBin
    from spacy.util import filter_spans

    try:
        nlp = spacy.load(args.base_model, disable=["ner", "parser", "tagger"])
    except Exception:
        nlp = spacy.blank("en")

    gold_docs = list(DocBin(store_user_data=True)
                     .from_disk(args.gold).get_docs(nlp.vocab))
    print(f"gold: {len(gold_docs)} docs, "
          f"{sum(len(d.ents) for d in gold_docs)} entities")
    print(f"model={args.model} passes={args.passes} temp={args.temperature} "
          f"ignore_case={args.ignore_case}\n")

    results = {}
    for variant in args.variants:
        print(f"--- variant: {variant} ---")
        tp = Counter(); fp = Counter(); fn = Counter()
        proposed_total = parse_fail = 0
        t0 = time.time()

        for doc in gold_docs:
            nid = str(doc.user_data.get("note_id", ""))
            gold_set = {(e.start_char, e.end_char, e.label_) for e in doc.ents}

            votes = Counter()
            for p in range(args.passes):
                try:
                    raw = call_ollama(PROMPTS[variant], doc.text, args.model,
                                      args.num_ctx, args.temperature)
                except (urllib.error.URLError, OSError) as exc:
                    print(f"  {nid} pass {p+1}: request failed ({exc})")
                    continue
                ents = parse_entities(raw)
                if ents is None:
                    parse_fail += 1
                    print(f"  {nid} pass {p+1}: parse failed")
                    continue
                votes.update(set(ents))
            proposals = {e for e, n in votes.items() if n >= args.min_votes}

            spans = []
            for txt, lab in proposals:
                for s, e in find_occurrences(doc.text, txt, args.ignore_case):
                    sp = doc.char_span(s, e, label=lab, alignment_mode="expand")
                    if sp is not None:
                        spans.append(sp)
            kept = filter_spans(spans)
            pred_set = {(sp.start_char, sp.end_char, sp.label_) for sp in kept}

            for k in gold_set & pred_set:
                tp[k[2]] += 1
            for k in pred_set - gold_set:
                fp[k[2]] += 1
            for k in gold_set - pred_set:
                fn[k[2]] += 1

            print(f"  {nid}: {len(proposals):>4} distinct proposals -> "
                  f"{len(kept):>4} located spans")

        print(f"\n  {'label':<12} {'P':>7} {'R':>7} {'F':>7}   "
              f"{'tp':>5} {'fp':>5} {'fn':>5}")
        for lab in sorted(VALID):
            p, r, f = prf(tp[lab], fp[lab], fn[lab])
            print(f"  {lab:<12} {p:7.2f} {r:7.2f} {f:7.2f}   "
                  f"{tp[lab]:5d} {fp[lab]:5d} {fn[lab]:5d}")
        P, R, F = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
        print(f"  {'OVERALL':<12} {P:7.2f} {R:7.2f} {F:7.2f}   "
              f"{sum(tp.values()):5d} {sum(fp.values()):5d} {sum(fn.values()):5d}")
        print(f"  ({time.time()-t0:.0f}s, {proposed_total} proposals, "
              f"{parse_fail} parse failures)\n")

        results[variant] = {"P": P, "R": R, "F": F, "passes": args.passes,
                            "proposals": proposed_total,
                            "parse_failures": parse_fail}

    print("=" * 62)
    print("BASELINE (production run, current prompt, 1 pass):")
    print("  OVERALL        75.94   35.25   48.15")
    print("=" * 62)
    for v, r in results.items():
        print(f"  {v:<14} {r['P']:7.2f} {r['R']:7.2f} {r['F']:7.2f}  "
              f"(x{r['passes']} passes)")

    if args.out_json:
        json.dump(results, open(args.out_json, "w"), indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
