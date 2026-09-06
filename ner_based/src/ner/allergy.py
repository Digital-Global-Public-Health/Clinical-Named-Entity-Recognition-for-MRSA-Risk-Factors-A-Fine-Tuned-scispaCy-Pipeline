"""Allergy detection for AIR·MS notes.

A drug named in an allergy list is a drug the patient did NOT receive. Left
alone it becomes a false positive for any medication feature, and for
`linezolid` / `vancomycin` a false positive on the leakage cluster
specifically. This module marks those spans so feature aggregation can drop
them.

Two mechanisms, one flag:

1. **Header regions** — a drug under an `Allergies:` header with no local cue.
   ConText cannot reach these: its scope is sentence-bounded, and the AIR·MS
   export has no line breaks, so an allergy table is one long "sentence" whose
   entries sit hundreds of characters from the header. Handled here by an
   explicit region rule, terminated by the next section header.

2. **Inline cues** — `allergic to vancomycin`, `PCN allergy`. Local and
   sentence-bounded, so this is ConText's job; added as ContextRules.

Both write `ent._.is_allergy`, with `ent._.allergy_source` recording which
fired. Nothing reads `span_attrs` for this, deliberately: passing a custom
`span_attrs` dict to ConText REPLACES medspaCy's defaults and would silently
disable is_negated / is_historical.

Measured basis (20,937-note silver corpus, 2026-08-20):
  1,330 notes (6.3 %) and 41 of 50 patients have >= 1 MEDICATION span under an
  allergy header. MAX_CHARS=300 chosen as a safety net for missed terminators:
  between 150 and 300 the span count moves 27 % but the note count moves 0.8 %,
  so the visit-level features this feeds are insensitive to the exact value.
"""

from __future__ import annotations

import re
from typing import Optional

from spacy.language import Language
from spacy.tokens import Span

# --------------------------------------------------------------------------
# Region detection
# --------------------------------------------------------------------------

# An allergy header at the start of the note or after a flattened line break
# (3+ spaces — the AIR·MS export contains no newlines), with up to two leading
# qualifier words ("Drug Allergies:", "Allergies / Intolerances").
ALLERGY_HDR = re.compile(
    r"(?:\A|\s{3,})(?:[A-Za-z/&]+[ ]){0,2}allerg(?:y|ies)\b[ ]*:?",
    re.I,
)

# What closes the region. Two shapes occur in this corpus:
#   - a capitalised label + colon, optionally numbered ("12. Local Anesthesia Given:")
#   - a medication-list header, which carries NO colon in the Epic export
#     ("Current Facility-Administered Medications")
NEXT_HDR = re.compile(
    r"\s{3,}(?:\d{1,2}[.)][ ]*)?[A-Z][A-Za-z0-9/&'-]{1,30}(?:[ ][A-Za-z0-9/&'-]{1,30}){0,5}:"
    r"|\s{3,}\S[^\r\n]{0,60}?[Mm]edications?\b"
)

MAX_CHARS = 300


NEGATED_HDR = re.compile(
    r"\b(no known (drug |medication )?allerg\w*|nkda|nkma|\bnka\b|"
    r"denies allerg\w*|allergies?\s*:?\s*(none|nil))\b",
    re.I)
NEGATED_HDR_WINDOW = 60


def governed_by_header(text: str, start: int, headers, window: int = MAX_CHARS):
    """Return the allergy header whose region contains `start`, or None.

    A region runs from the header to whichever comes first: the next section
    header, or `window` characters.
    """
    best = None
    for h in headers:
        if h.end() <= start:
            best = h
        else:
            break
    if best is None:
        return None
    if start - best.end() > window:
        return None
    if NEXT_HDR.search(text, best.end(), start):
        return None
    # An allergy header that immediately negates itself governs nothing.
    # "No Known Allergies   Scheduled Medications: ..." otherwise opens a
    # 300-char region over the medication list and flags every drug in it
    # as an allergen. In a 40-span adjudicated corpus sample this was 12 of
    # 14 false positives -- the dominant failure mode of the header path,
    # not the rare residual it was previously recorded as. Note the ConText
    # pseudo-modifiers only suppress the inline-cue path, never this one.
    # ALLERGY_HDR allows up to two words before "allergies", so the match
    # itself swallows the negation: "No Known Allergies" IS the header.
    # Check the matched text, then the region that follows it.
    if NEGATED_HDR.search(best.group()):
        return None
    if NEGATED_HDR.search(text, best.end(),
                          best.end() + NEGATED_HDR_WINDOW):
        return None
    return best


def find_allergy_headers(text: str):
    return list(ALLERGY_HDR.finditer(text))


# --------------------------------------------------------------------------
# Inline cues (ConText)
# --------------------------------------------------------------------------

def build_allergy_rules():
    """ContextRules for inline allergy cues, plus pseudo-modifiers.

    The pseudo rules exist because `allergic` is also a disease adjective:
    `allergic rhinitis` is a diagnosis, not an allergy to rhinitis. They are
    longer literals in a category nothing consumes, so ConText's
    prune_on_modifier_overlap discards the shorter real cue underneath.
    """
    from medspacy.context import ConTextRule

    forward = [
        "allergic to", "allergy to", "allergies to", "allergic reaction to",
        "intolerant to", "intolerance to", "adverse reaction to",
        "adverse drug reaction to", "sensitivity to", "reaction to",
    ]
    # Backward cues must be capped or `penicillin allergy` reaches back over
    # every drug earlier in the sentence.
    backward = ["allergy", "allergies", "intolerance", "adverse reaction"]
    pseudo = [
        "allergic rhinitis", "allergic asthma", "allergic conjunctivitis",
        "allergic reaction", "seasonal allergies", "environmental allergies",
        "allergy testing", "allergy test", "allergy shots", "allergy clinic",
        "no known allergies", "no known drug allergies", "denies allergies",
        "no allergies", "allergies reviewed", "allergies: none",
    ]

    rules = [ConTextRule(lit, "ALLERGY", direction="FORWARD", max_scope=6)
             for lit in forward]
    rules += [ConTextRule(lit, "ALLERGY", direction="BACKWARD",
                          max_scope=3, max_targets=1) for lit in backward]
    rules += [ConTextRule(lit, "PSEUDO_ALLERGY", direction="BIDIRECTIONAL",
                          max_scope=1) for lit in pseudo]
    return rules


def add_allergy_rules(nlp) -> int:
    """Push the inline cues into an existing medspacy_context pipe."""
    ctx = nlp.get_pipe("medspacy_context")
    rules = build_allergy_rules()
    ctx.add(rules)
    return len(rules)


# --------------------------------------------------------------------------
# The component
# --------------------------------------------------------------------------

for _attr, _default in (("is_allergy", False), ("allergy_source", "")):
    if not Span.has_extension(_attr):
        Span.set_extension(_attr, default=_default)


@Language.factory(
    "airms_allergy",
    default_config={"max_chars": MAX_CHARS, "labels": ["MEDICATION"]},
)
def create_airms_allergy(nlp, name: str, max_chars: int, labels):
    return AirmsAllergy(max_chars=max_chars, labels=labels)


class AirmsAllergy:
    """Set `ent._.is_allergy` from header regions and from ConText cues.

    Place AFTER medspacy_context so the cue modifiers are already attached.
    """

    def __init__(self, max_chars: int = MAX_CHARS, labels=("MEDICATION",)):
        self.max_chars = max_chars
        self.labels = set(labels)

    def __call__(self, doc):
        headers = find_allergy_headers(doc.text)
        for ent in doc.ents:
            if ent.label_ not in self.labels:
                continue
            by_header = governed_by_header(
                doc.text, ent.start_char, headers, self.max_chars) is not None
            by_cue = any(
                getattr(m, "category", None) == "ALLERGY"
                for m in (ent._.modifiers or ())
            ) if Span.has_extension("modifiers") else False

            if by_header or by_cue:
                ent._.is_allergy = True
                ent._.allergy_source = "+".join(
                    s for s, on in (("header", by_header), ("cue", by_cue)) if on
                )
        return doc


def is_administered(ent) -> bool:
    """The predicate feature aggregation should use for MEDICATION spans.

    An allergy is not an administration. Neither is a negated mention — and
    note that `no penicillin allergy` is BOTH: it says the allergy does not
    exist, which is still no evidence the drug was given. Both drop.
    """
    if ent._.is_allergy:
        return False
    for attr in ("is_negated", "is_hypothetical", "is_family"):
        if Span.has_extension(attr) and getattr(ent._, attr):
            return False
    return True
