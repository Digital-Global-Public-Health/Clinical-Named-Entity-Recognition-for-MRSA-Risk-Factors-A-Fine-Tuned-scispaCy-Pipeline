#!/usr/bin/env python
"""Constrain two of medspaCy's packaged ConText rules (2026-08-26).

Found while validating the feature matrix on 200 notes: 16 of 34 `cefepime`
mentions were being marked negated, all of them wrongly -- `Continue Cefepime`,
`Agree with Cefepime and Vancomycin`, and plain MAR entries.

Cause: two PACKAGED default rules, not AIR.MS rules (the pipeline loads 171
rules, ~140 of them medspaCy defaults).

  'prophylaxis'  NEGATED_EXISTENCE  BACKWARD  max_scope=None  allowed=None
      Fires backward over any preceding entity. In
      `Continue Cefepime   Prophylaxis: acyclovir` it negates the cefepime.
      But prophylaxis never negates the DRUG -- the drug is given. It negates
      the CONDITION being prevented, which is exactly what annotation
      guideline 3.4 says (`DVT PPx` -> tag Lovenox, not DVT).
      Fix: restrict to DISEASE targets. `DVT prophylaxis` still negates DVT.

  ': no'         NEGATED_EXISTENCE  BACKWARD  max_scope=None  allowed=None
      Observed reaching >500 characters backward, across several sections.
      Fix: cap max_scope.

Both are mutated in place after load; medspaCy reads allowed_types and
max_scope from the rule object at modify time, so this takes effect without
rebuilding the matcher (verified).

Impact if left unfixed: a systematic under-count across all 17 MEDICATION
features, since antibiotic / GI / DVT prophylaxis appears in nearly every
inpatient note. Invisible in the finished matrix.

Run from ner_based/.
"""

import pathlib
import sys

p = pathlib.Path("src/ner/assertion.py")
src = p.read_text()

old = """    context = nlp.get_pipe("medspacy_context")
    context.add(AIRMS_CONTEXT_RULES)
"""
new = '''    context = nlp.get_pipe("medspacy_context")
    _constrain_packaged_rules(context)
    context.add(AIRMS_CONTEXT_RULES)
'''
if src.count(old) != 1:
    sys.exit(f"! anchor matched {src.count(old)} times, aborting")
src = src.replace(old, new)

helper = '''
# Two of medspaCy's packaged rules mis-fire on this corpus. See
# scripts/patch_assertion_packaged_rules.py for the evidence.
PACKAGED_RULE_FIXES = {
    # Prophylaxis negates the condition being prevented, never the drug.
    # Restricting to DISEASE keeps `DVT prophylaxis` -> DVT negated, while
    # `Continue Cefepime   Prophylaxis: acyclovir` no longer negates cefepime.
    "prophylaxis": {"allowed_types": {"DISEASE"}},
    # Observed reaching >500 chars backward across section boundaries.
    ": no": {"max_scope": 5},
}


def _constrain_packaged_rules(context) -> int:
    """Tighten packaged ConText rules in place. Returns the number changed."""
    changed = 0
    for rule in context.rules:
        fix = PACKAGED_RULE_FIXES.get((rule.literal or "").lower())
        if not fix:
            continue
        for attr, value in fix.items():
            setattr(rule, attr, value)
        changed += 1
    return changed


'''

anchor = "def build_assertion_pipeline("
if src.count(anchor) != 1:
    sys.exit(f"! build_assertion_pipeline anchor matched {src.count(anchor)} times")
src = src.replace(anchor, helper.lstrip("\n") + anchor)

p.write_text(src)
print("patched src/ner/assertion.py")
