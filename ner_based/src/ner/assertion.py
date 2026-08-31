"""Clinical assertion detection layered on top of the AIR.MS NER model.

Gold NER annotations remain span-and-label only. This module applies medspaCy
ConText after NER at inference/review time and exposes the resulting assertion,
temporality, and experiencer flags on each predicted entity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Tuple, Union

import medspacy
from loguru import logger
from medspacy.context import ConTextRule
from spacy.language import Language
from .allergy import add_allergy_rules

logger.disable("PyRuSH")  # PyRuSH emits ~6 DEBUG lines per sentence

ALLOWED_LABELS = frozenset({"DISEASE", "MEDICATION", "PROCEDURE"})

# Path to the custom spaCy registrations required by the trained model config.
CUSTOM_CODE_PATH = Path(__file__).resolve().parents[2] / "configs" / "custom_code.py"

# AIR.MS-specific additions to medspaCy's default ConText rules. Each comment
# records why the cue is included and the direction in which it modifies an
# entity. Matching is case-insensitive under ConText's default LOWER matcher.
#
# Category -> attribute mapping (medspaCy defaults):
#   NEGATED_EXISTENCE  -> ent._.is_negated
#   HISTORICAL         -> ent._.is_historical
#   HYPOTHETICAL       -> ent._.is_hypothetical   (conditional / future)
#   POSSIBLE_EXISTENCE -> ent._.is_uncertain      (suspected / differential)
#   FAMILY             -> ent._.is_family
# Several of these cues duplicate medspaCy's packaged defaults (loaded via
# load_rules=True). Duplicate matches are idempotent for boolean flags; the
# rules are restated here so the AIR.MS cue set is explicit and auditable.
AIRMS_CONTEXT_RULES = [
    # "no": the most common compact pre-negation in problem and review lists.
    ConTextRule("no", "NEGATED_EXISTENCE", direction="FORWARD"),
    # "denies": patient denial negates the following symptom or condition.
    ConTextRule("denies", "NEGATED_EXISTENCE", direction="FORWARD"),
    # "without": absence phrased as a preposition modifies the following entity.
    ConTextRule("without", "NEGATED_EXISTENCE", direction="FORWARD"),
    # "negative for": test/review language negates the following finding.
    ConTextRule("negative for", "NEGATED_EXISTENCE", direction="FORWARD"),
    # "not": broad clinical shorthand used immediately before an absent finding.
    ConTextRule("not", "NEGATED_EXISTENCE", direction="FORWARD"),
    # "no evidence of": multi-token absence cue used in assessments and imaging.
    ConTextRule("no evidence of", "NEGATED_EXISTENCE", direction="FORWARD"),
    # "free of": states that the following condition is absent.
    ConTextRule("free of", "NEGATED_EXISTENCE", direction="FORWARD"),
    # "rules out": completed diagnostic language is treated as negation.
    ConTextRule("rules out", "NEGATED_EXISTENCE", direction="FORWARD"),
    # "r/o": mapped to negation per the AIR.MS specification; in some notes it
    # instead means an unresolved differential, so this cue requires review.
    ConTextRule("r/o", "POSSIBLE_EXISTENCE", direction="FORWARD"),
    # "h/o": standard abbreviation introducing past medical history.
    ConTextRule("h/o", "HISTORICAL", direction="FORWARD"),
    # "hx of": compact history phrase modifying the following entity.
    ConTextRule("hx of", "HISTORICAL", direction="FORWARD"),
    # "PMHx": past-medical-history section shorthand with forward scope.
    ConTextRule("PMHx", "HISTORICAL", direction="FORWARD"),
    # "PMH": past-medical-history section shorthand with forward scope.
    ConTextRule("PMH", "HISTORICAL", direction="FORWARD"),
    # "s/p": status-post abbreviation indicating a prior procedure/event.
    ConTextRule("s/p", "HISTORICAL", direction="FORWARD"),
    # "status post": expanded status-post phrase for a prior procedure/event.
    ConTextRule("status post", "HISTORICAL", direction="FORWARD"),
    # "history of": explicit phrase introducing a historical entity.
    ConTextRule("history of", "HISTORICAL", direction="FORWARD"),
    # "previous": adjective marking the following entity as historical.
    ConTextRule("previous", "HISTORICAL", direction="FORWARD"),
    # "prior": adjective marking the following entity as historical.
    ConTextRule("prior", "HISTORICAL", direction="FORWARD"),
    # "if": conditional plans make subsequent mentions hypothetical.
    ConTextRule("if", "HYPOTHETICAL", direction="FORWARD"),
    # "should": conditional recommendation/planning cue with forward scope.
    ConTextRule("should", "HYPOTHETICAL", direction="FORWARD"),
    # "concern for": introduces a suspected rather than established condition.
    ConTextRule("concern for", "POSSIBLE_EXISTENCE", direction="FORWARD"),
    # "c/f": compact form of "concern for".
    ConTextRule("c/f", "POSSIBLE_EXISTENCE", direction="FORWARD"),
    # "possible": explicitly marks the following entity as uncertain.
    ConTextRule("possible", "POSSIBLE_EXISTENCE", direction="FORWARD"),
    # "may represent": interpretation language marking the next entity uncertain.
    ConTextRule("may represent", "POSSIBLE_EXISTENCE", direction="FORWARD"),
    # "versus": both sides of a differential are unconfirmed alternatives.
    ConTextRule("versus", "POSSIBLE_EXISTENCE", direction="BIDIRECTIONAL"),
    # "vs": abbreviated differential; like "versus", it modifies both sides.
    ConTextRule("vs", "POSSIBLE_EXISTENCE", direction="BIDIRECTIONAL"),
    # "rule out": an unresolved diagnostic instruction, not a confirmed absence.
    ConTextRule("rule out", "POSSIBLE_EXISTENCE", direction="FORWARD"),
    # "consider": plan/assessment language makes the following entity provisional.
    ConTextRule("consider", "HYPOTHETICAL", direction="FORWARD"),
    # "mother": assigns the following clinical mention to another experiencer.
    ConTextRule("mother", "FAMILY", direction="FORWARD"),
    # "father": assigns the following clinical mention to another experiencer.
    ConTextRule("father", "FAMILY", direction="FORWARD"),
    # "sister": assigns the following clinical mention to another experiencer.
    ConTextRule("sister", "FAMILY", direction="FORWARD"),
    # "brother": assigns the following clinical mention to another experiencer.
    ConTextRule("brother", "FAMILY", direction="FORWARD"),
    # "family history": explicit family-experiencer phrase.
    ConTextRule("family history", "FAMILY", direction="FORWARD"),
    # "FHx": compact family-history section shorthand.
    ConTextRule("FHx", "FAMILY", direction="FORWARD"),
    # Coordinating contrast ends preceding ConText scope.
    # --- hypothetical cue families, added 2026-08-31 -------------------------
    # Derived from the 16-note gold set: the axis previously had only "if" and
    # "should" and scored 3 TP / 38 FN. 26 of the 38 misses were PROCEDURE, so
    # planned and pending studies were entering the feature layer as performed.
    # NOTE: these cues were developed against the gold set, so any F1 reported
    # for the hypothetical axis is an upper bound, not a held-out estimate.

    # Planned interventions: the entity is proposed, not performed.
    ConTextRule("plan for", "HYPOTHETICAL", direction="FORWARD"),
    ConTextRule("plan is to", "HYPOTHETICAL", direction="FORWARD"),
    ConTextRule("planned for", "HYPOTHETICAL", direction="FORWARD"),
    ConTextRule("will need", "HYPOTHETICAL", direction="FORWARD"),
    ConTextRule("needs", "HYPOTHETICAL", direction="FORWARD"),
    # Proposals from a consulting service. Guideline B5 treats these as
    # hypothetical: the note records the proposal, not the administration.
    ConTextRule("recommend", "HYPOTHETICAL", direction="FORWARD"),
    ConTextRule("recommends", "HYPOTHETICAL", direction="FORWARD"),
    ConTextRule("recommending", "HYPOTHETICAL", direction="FORWARD"),
    ConTextRule("suggest", "HYPOTHETICAL", direction="FORWARD"),
    # Watch-for lists name side effects to anticipate, not present findings.
    ConTextRule("monitor for", "HYPOTHETICAL", direction="FORWARD"),
    # Conditional administration: "hold for active bleeding".
    ConTextRule("hold for", "HYPOTHETICAL", direction="FORWARD"),
    # Study purpose: the entity is what is being looked for.
    ConTextRule("for detection of", "HYPOTHETICAL", direction="FORWARD"),
    # Postpositive in this corpus -- "CT chest pending", "MRI pending" --
    # so these scope backward onto the preceding study, not forward.
    ConTextRule("pending", "HYPOTHETICAL", direction="BACKWARD"),
    ConTextRule("ordered", "HYPOTHETICAL", direction="BACKWARD"),
    # --- end hypothetical additions ------------------------------------------

    ConTextRule("but", "TERMINATE", direction="TERMINATE"),
    # Sentence-level contrast likewise starts a new assertion scope.
    ConTextRule("however", "TERMINATE", direction="TERMINATE"),
    # Semicolons commonly separate independent findings in clinical prose.
    ConTextRule(";", "TERMINATE", direction="TERMINATE"),
    # AIR.MS flattened every line break to spaces (20,937 notes have no newlines);
    # spaCy represents a 3+-space boundary as one whitespace token of length >= 2.
    # These runs preserve section-header and list boundaries for scope termination.
    ConTextRule(
        "   ",
        "TERMINATE",
        direction="TERMINATE",
        pattern=[{"IS_SPACE": True, "LENGTH": {">=": 2}}],
    ),
    # Bullet characters survive export flattening and separate problem-list
    # entries, so they terminate scope between otherwise adjacent entries.
    ConTextRule(
        "•",
        "TERMINATE",
        direction="TERMINATE",
        pattern=[{"TEXT": "•"}],
    ),
]

TextInput = Union[str, Tuple[str, str], Mapping[str, Any]]


def _load_custom_code(path: Path = CUSTOM_CODE_PATH) -> None:
    """Register the project's custom spaCy functions before loading a model.

    The trained model config references ``airms.scispacy_tokenizer.v1``, a
    callback registered in ``configs/custom_code.py``. ``spacy train`` and
    ``spacy evaluate`` see it via ``--code``; a plain ``spacy.load`` does not,
    and fails with ``catalogue.RegistryError [E893]``. Importing the module for
    its registration side effects is what makes model loading work here.

    NOT UNUSED -- do not remove as dead code.
    """
    if not path.exists():
        raise FileNotFoundError(f"custom code module not found: {path}")
    spec = importlib.util.spec_from_file_location("airms_custom_code", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load custom code module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


# Two of medspaCy's packaged rules mis-fire on this corpus. See
# scripts/patch_assertion_packaged_rules.py for the evidence.
PACKAGED_RULE_FIXES = {
    # Prophylaxis negates the condition being prevented, never the drug.
    # Restricting to DISEASE keeps `DVT prophylaxis` -> DVT negated, while
    # `Continue Cefepime   Prophylaxis: acyclovir` no longer negates cefepime.
    "prophylaxis": {"allowed_types": {"DISEASE"}},
    # Observed reaching >500 chars backward across section boundaries.
    ": no": {"max_scope": 5},
    # Guideline B6: a PRN indication is affirmed, not hypothetical --
    # "as needed for Cough" prescribes the drug; it does not hypothesise
    # the cough. For some patients the medication-list indication is the
    # only place a symptom is documented, so hypothesising it would make
    # the symptom invisible to the feature layer. Restricting scope to 0
    # disables the rule while leaving the decision visible here.
    "as needed": {"max_scope": 0},
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


def build_assertion_pipeline(model_path: Union[str, Path]) -> Language:
    """Load a fine-tuned AIR.MS NER model and append medspaCy ConText.

    medspaCy's standard ConText rules are loaded first, after which the
    AIR.MS-specific rules above are added. The trained models contain no parser
    or sentence segmenter, so PyRuSH is inserted *after* NER: ConText scopes
    over sentences and needs boundaries, while inserting a component before NER
    would risk perturbing the entity predictions the model was evaluated on.
    PyRuSH is used rather than spaCy's punctuation-based ``sentencizer`` because
    clinical notes contain problem lists, bulleted findings and headers with no
    terminal punctuation, where a whole section would otherwise collapse into a
    single sentence and let one negation cue scope over all of it.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"NER model path does not exist: {model_path}")

    # Must precede any model load; see _load_custom_code.
    _load_custom_code()

    # medspaCy delegates model loading to spaCy, keeps the fine-tuned pipeline,
    # and appends only ConText (with its packaged default rules).
    nlp = medspacy.load(
        str(model_path),
        medspacy_enable=["medspacy_context"],
        load_rules=True,
    )
    if "ner" not in nlp.pipe_names:
        raise ValueError(f"spaCy model has no NER component: {model_path}")
    sentence_components = {"parser", "senter", "sentencizer", "medspacy_pyrush"}
    if not sentence_components.intersection(nlp.pipe_names):
        nlp.add_pipe("medspacy_pyrush", after="ner")

    # Section detection: assigns ent._.section_category as descriptive metadata
    # only. Nothing downstream reads it -- section *boundaries* are unreliable
    # on this corpus, because the export has no line breaks and a region only
    # closes when the next header is matched. The one section that matters is
    # handled explicitly by airms_allergy instead. Do not enable add_attrs.
    if "medspacy_sectionizer" not in nlp.pipe_names:
        nlp.add_pipe("medspacy_sectionizer", before="medspacy_context")

    context = nlp.get_pipe("medspacy_context")
    _constrain_packaged_rules(context)
    context.add(AIRMS_CONTEXT_RULES)
    add_allergy_rules(nlp)
    if "airms_allergy" not in nlp.pipe_names:
        nlp.add_pipe("airms_allergy")
    if nlp.pipe_names.index("medspacy_context") <= nlp.pipe_names.index("ner"):
        raise RuntimeError("medspacy_context must run after the NER component")
    return nlp


def annotate_assertions(
    nlp: Language,
    texts: Iterable[TextInput],
    *,
    batch_size: int = 32,
) -> Iterator[Dict[str, Any]]:
    """Yield one assertion record per predicted clinical entity.

    ``texts`` may contain plain strings, ``(note_id, text)`` pairs, or mappings
    with ``note_id`` and ``text`` keys. Plain strings receive their zero-based
    input position as ``note_id``. Only the three annotation-schema labels are
    emitted.
    """
    if "ner" not in nlp.pipe_names or "medspacy_context" not in nlp.pipe_names:
        raise ValueError("nlp must contain NER followed by medspacy_context")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    stream = _normalise_text_inputs(texts)
    for doc, note_id in nlp.pipe(stream, as_tuples=True, batch_size=batch_size):
        for ent in doc.ents:
            if ent.label_ not in ALLOWED_LABELS:
                continue
            yield {
                "note_id": note_id,
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "is_negated": bool(getattr(ent._, "is_negated", False)),
                "is_historical": bool(getattr(ent._, "is_historical", False)),
                "is_hypothetical": bool(getattr(ent._, "is_hypothetical", False)),
                "is_uncertain": bool(getattr(ent._, "is_uncertain", False)),
                "is_family": bool(getattr(ent._, "is_family", False)),
            }


def _normalise_text_inputs(texts: Iterable[TextInput]) -> Iterator[Tuple[str, str]]:
    """Convert supported caller inputs to spaCy's ``(text, context)`` form."""
    for index, item in enumerate(texts):
        if isinstance(item, str):
            yield item, str(index)
            continue
        if isinstance(item, Mapping):
            if "note_id" not in item or "text" not in item:
                raise ValueError("text mappings must contain note_id and text")
            yield str(item["text"]), str(item["note_id"])
            continue
        try:
            note_id, text = item
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "texts must contain strings, (note_id, text) pairs, or mappings"
            ) from exc
        yield str(text), str(note_id)