# src/ner/annotation_schema.py
"""
Entity annotation schema for the MRSA NLP NER pipeline.

Defines the entity types used for annotation and NER model training:
  - DISEASE     : clinical conditions, diagnoses, symptoms
  - MEDICATION  : drugs, drug classes, pharmacological agents
  - PROCEDURE   : clinical procedures, devices, interventions
  - SEVERITY    : clinical severity / immune-status indicators (optional)

This module also provides export utilities to write annotation guidelines
that annotators can follow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

LOG = logging.getLogger("mrsa_nlp.ner.schema")


# ---------------------------------------------------------------------------
# Entity type definitions
# ---------------------------------------------------------------------------

@dataclass
class EntityType:
    """
    Describes one NER entity type.

    Attributes
    ----------
    name : str
        Short label used in annotations (e.g. "DISEASE").
    description : str
        Plain-language definition.
    examples : list of str
        Representative entity surface forms.
    exclusions : list of str
        Edge cases that should NOT be tagged with this label.
    annotation_tips : list of str
        Practical guidance for human annotators.
    mrsa_relevance : str
        Why entities of this type are relevant for MRSA prediction.
    """

    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    annotation_tips: List[str] = field(default_factory=list)
    mrsa_relevance: str = ""


# ---------------------------------------------------------------------------
# Default MRSA-relevant entity schema
# ---------------------------------------------------------------------------

MRSA_ENTITY_TYPES: Dict[str, EntityType] = {
    "DISEASE": EntityType(
        name="DISEASE",
        description=(
            "Any clinical condition, diagnosis, chronic illness, infection, or symptom "
            "mentioned as present in the patient."
        ),
        examples=[
            "pneumonia", "MRSA", "methicillin-resistant Staphylococcus aureus",
            "UTI", "urinary tract infection", "sepsis", "bacteremia",
            "diabetes mellitus", "end-stage renal disease", "rheumatoid arthritis",
            "HIV", "lymphoma", "cellulitis", "fever", "abscess",
        ],
        exclusions=[
            "Negated conditions: 'no pneumonia', 'denied fever'",
            "Rule-out: 'rule out sepsis' — use judgement based on context",
            "Historical: 'history of' should still be tagged if clinically relevant",
            "Allergies: 'penicillin allergy' — the disease 'allergy' may be tagged but "
            "the allergen itself (penicillin) should be tagged MEDICATION with an allergy flag",
        ],
        annotation_tips=[
            "Include the full span: 'end-stage renal disease' not just 'disease'",
            "Tag multi-word conditions as a single span",
            "Do NOT tag lab values or vital signs as DISEASE",
        ],
        mrsa_relevance=(
            "Comorbidities (CKD, diabetes, immunosuppressive diseases) and prior infections "
            "(prior MRSA, prior bacteremia) are key MRSA risk factors."
        ),
    ),

    "MEDICATION": EntityType(
        name="MEDICATION",
        description=(
            "Any drug, pharmacological agent, antibiotic, immunosuppressant, corticosteroid, "
            "or drug class mentioned in the note."
        ),
        examples=[
            "prednisone", "methylprednisolone", "dexamethasone",
            "vancomycin", "methotrexate", "azathioprine",
            "mycophenolate mofetil", "cyclosporine", "tacrolimus",
            "corticosteroids", "antibiotics", "NSAIDs", "penicillin",
            "ciprofloxacin", "piperacillin-tazobactam",
        ],
        exclusions=[
            "Allergies: tag 'penicillin' in 'penicillin allergy' but note the allergy context",
            "Do NOT tag vitamin supplements as MEDICATION unless clinically significant",
            "Do NOT tag food/diet",
        ],
        annotation_tips=[
            "Tag both brand and generic names",
            "Tag drug classes when specific drugs are not named",
            "Dose/frequency ('40 mg daily') is NOT part of the MEDICATION span",
        ],
        mrsa_relevance=(
            "Immunosuppressants (corticosteroids, anti-rejection drugs) are direct MRSA risk factors. "
            "Prior antibiotic exposure (especially broad-spectrum) selects for resistant organisms."
        ),
    ),

    "PROCEDURE": EntityType(
        name="PROCEDURE",
        description=(
            "Any clinical procedure, surgical intervention, invasive device, or diagnostic test "
            "the patient underwent or currently has in place."
        ),
        examples=[
            "central line", "central venous catheter", "PICC line",
            "hemodialysis", "peritoneal dialysis", "intubation",
            "mechanical ventilation", "tracheostomy",
            "surgery", "coronary artery bypass graft", "organ transplant",
            "bone marrow transplant", "wound debridement",
            "chest tube", "Foley catheter",
        ],
        exclusions=[
            "Lab draws (CBC, blood culture) unless an invasive procedure is explicitly described",
            "Imaging (X-ray, CT) unless directly involving invasive access",
        ],
        annotation_tips=[
            "Focus on procedures that breach skin or mucosal barriers",
            "Device presence (e.g. 'patient has a central line') should be tagged",
            "Include the full device name: 'peripherally inserted central catheter' not 'catheter'",
        ],
        mrsa_relevance=(
            "Invasive procedures and long-term devices are primary MRSA acquisition routes "
            "(central line-associated bloodstream infections, dialysis access infections)."
        ),
    ),

    "SEVERITY": EntityType(
        name="SEVERITY",
        description=(
            "Terms indicating the severity of a clinical state, immune competence, or "
            "overall patient vulnerability relevant to infection risk.  This is an optional "
            "advanced entity type."
        ),
        examples=[
            "critically ill", "immunocompromised", "immunosuppressed",
            "neutropenic", "severely immunodeficient",
            "debilitated", "malnourished", "cachexic",
        ],
        exclusions=[
            "Generic severity words without clinical specificity: 'severe pain'",
            "Vital-sign descriptions: 'hypotensive', 'febrile'",
        ],
        annotation_tips=[
            "Use sparingly; only when the term clearly indicates systemic immune status",
            "Can overlap with DISEASE (e.g. 'neutropenic fever' — tag 'neutropenic' as SEVERITY)",
        ],
        mrsa_relevance=(
            "Immune status is a cross-cutting risk modifier; immunocompromised patients "
            "have disproportionately high MRSA bacteremia risk."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnnotationSchemaConfig:
    """
    Parameters for the annotation schema.

    Attributes
    ----------
    entity_types : list of str
        Entity type labels to use.  Must be keys of MRSA_ENTITY_TYPES.
        Defaults to core types; add "SEVERITY" for advanced annotation.
    guidelines_out_path : Path
        Where to write the guidelines Markdown document.
    debug : bool
    """

    entity_types: List[str] = field(
        default_factory=lambda: ["DISEASE", "MEDICATION", "PROCEDURE"]
    )
    guidelines_out_path: Path = Path("annotations/schema_reference.md")
    debug: bool = False


# ---------------------------------------------------------------------------
# Schema class
# ---------------------------------------------------------------------------

class AnnotationSchema:
    """
    Manages the NER entity type schema and annotation guidelines.

    Parameters
    ----------
    config : AnnotationSchemaConfig
    logger : logging.Logger, optional

    Example
    -------
    >>> schema = AnnotationSchema(AnnotationSchemaConfig())
    >>> schema.export_guidelines()
    >>> labels = schema.get_entity_labels()
    """

    def __init__(
        self,
        config: AnnotationSchemaConfig,
        logger: logging.Logger = LOG,
    ) -> None:
        self.cfg = config
        self.log = logger

    def get_entity_labels(self) -> List[str]:
        """
        Return the list of entity type labels configured for this schema.

        Returns
        -------
        list of str
            E.g. ["DISEASE", "MEDICATION", "PROCEDURE"].
        """
        labels = list(self.cfg.entity_types)
        unknown = [label for label in labels if label not in MRSA_ENTITY_TYPES]
        if unknown:
            raise KeyError(f"Unknown entity type(s): {unknown}")
        return labels

    def get_entity_definition(self, entity_type: str) -> EntityType:
        """
        Return the EntityType definition for a given label.

        Parameters
        ----------
        entity_type : str
            Must be a key in MRSA_ENTITY_TYPES.

        Returns
        -------
        EntityType

        Raises
        ------
        KeyError
            If entity_type is not defined.
        """
        if entity_type not in MRSA_ENTITY_TYPES:
            raise KeyError(f"Unknown entity type: {entity_type}")
        return MRSA_ENTITY_TYPES[entity_type]

    def validate_annotation(
        self,
        text: str,
        entities: List[dict],
    ) -> List[str]:
        """
        Validate a set of entity annotations against the schema rules.

        Parameters
        ----------
        text : str
            The note text being annotated.
        entities : list of dict
            Each dict: {"start": int, "end": int, "label": str, "text": str}.

        Returns
        -------
        list of str
            List of validation warning messages.  Empty list = no warnings.

        Notes
        -----
        Checks:
        - All labels are in cfg.entity_types.
        - Span [start, end) is a valid substring of text.
        - No duplicate spans (same start/end with same label).
        - No overlapping spans.
        - Span text is non-empty.
        """
        warnings: List[str] = []
        labels = set(self.get_entity_labels())
        seen = set()
        spans = []

        for ent in entities:
            start = ent.get("start")
            end = ent.get("end")
            label = ent.get("label")
            surface = ent.get("text", "")

            if label not in labels:
                warnings.append(f"Unknown label {label!r} for span {start}:{end}")
            if not isinstance(start, int) or not isinstance(end, int):
                warnings.append(f"Span offsets must be integers: {ent}")
                continue
            if start < 0 or end > len(text) or start >= end:
                warnings.append(f"Invalid span offsets {start}:{end}")
                continue
            if not text[start:end].strip():
                warnings.append(f"Empty or whitespace-only span {start}:{end}")
            if surface and text[start:end] != surface:
                warnings.append(
                    f"Span text mismatch at {start}:{end}: expected {surface!r}, got {text[start:end]!r}"
                )

            key = (start, end, label)
            if key in seen:
                warnings.append(f"Duplicate span {start}:{end}:{label}")
            seen.add(key)
            spans.append((start, end, label))

        for i, (start_a, end_a, label_a) in enumerate(spans):
            for start_b, end_b, label_b in spans[i + 1 :]:
                if start_a < end_b and start_b < end_a:
                    warnings.append(
                        f"Overlapping spans {start_a}:{end_a}:{label_a} and {start_b}:{end_b}:{label_b}"
                    )

        return warnings

    def export_guidelines(self) -> Path:
        """
        Write a Markdown annotation guidelines document to cfg.guidelines_out_path.

        The document includes:
        - Introduction and purpose
        - General annotation rules (negation, speculation, historical mentions)
        - Per-entity-type sections: definition, examples, exclusions, tips,
          MRSA relevance
        - Inter-annotator agreement section
        - Edge-cases FAQ

        Returns
        -------
        Path
            Path to the written guidelines file.

        Notes
        -----
        - Create the parent directory if needed.
        - Log the output path.
        """
        self.cfg.guidelines_out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# MRSA NER Entity Schema Reference",
            "",
            "Use `CONTRACT.md` as the source of truth for training-data format.",
            "",
            "## Entity Labels",
            "",
        ]
        for label in self.get_entity_labels():
            definition = self.get_entity_definition(label)
            lines.extend(
                [
                    f"### {label}",
                    "",
                    definition.description,
                    "",
                    "Examples: " + ", ".join(f"`{item}`" for item in definition.examples),
                    "",
                    "Exclusions: " + "; ".join(definition.exclusions),
                    "",
                    "Tips: " + "; ".join(definition.annotation_tips),
                    "",
                ]
            )
        self.cfg.guidelines_out_path.write_text("\n".join(lines) + "\n")
        self.log.info("Annotation guidelines written to %s", self.cfg.guidelines_out_path)
        return self.cfg.guidelines_out_path

    def to_spacy_labels(self) -> List[str]:
        """
        Return entity labels formatted for spaCy NER component registration.

        Returns
        -------
        list of str
            Same as get_entity_labels() — spaCy uses plain string labels.
        """
        return self.get_entity_labels()

    def export_schema_json(self, out_path: Optional[Path] = None) -> Path:
        """
        Write the schema as a JSON file for use by annotation tools.

        Parameters
        ----------
        out_path : Path, optional
            Defaults to ``annotations/schema.json``.

        Returns
        -------
        Path
            Written file path.
        """
        if out_path is None:
            out_path = Path("annotations/schema.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entity_types": [
                {
                    "name": entity.name,
                    "description": entity.description,
                    "examples": entity.examples,
                    "exclusions": entity.exclusions,
                    "annotation_tips": entity.annotation_tips,
                    "mrsa_relevance": entity.mrsa_relevance,
                }
                for entity in (self.get_entity_definition(label) for label in self.get_entity_labels())
            ]
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        self.log.info("Annotation schema JSON written to %s", out_path)
        return out_path
