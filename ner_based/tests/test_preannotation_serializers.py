from __future__ import annotations

import importlib.util
import unittest

from src.ner.preannotation_serializers import SerializerSummary, _artifact_to_webanno_tsv


SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None


def _token_rows(tsv_text: str) -> list[list[str]]:
    return [
        line.split("\t")
        for line in tsv_text.splitlines()
        if line and not line.startswith("#")
    ]


@unittest.skipUnless(SPACY_AVAILABLE, "spaCy is required for WebAnno TSV serialization")
class WebAnnoSerializerTest(unittest.TestCase):
    def test_period_is_separate_token_and_span_covers_entity_tokens(self) -> None:
        text = "Patient has diabetes mellitus."
        start = text.index("diabetes mellitus")
        end = start + len("diabetes mellitus")
        summary = SerializerSummary()

        tsv = _artifact_to_webanno_tsv(
            text,
            [
                {
                    "text": "diabetes mellitus",
                    "start_char": start,
                    "end_char": end,
                    "label": "DISEASE",
                }
            ],
            note_id="TSV-N001",
            summary=summary,
        )
        rows = _token_rows(tsv)

        self.assertEqual([row[2] for row in rows], ["Patient", "has", "diabetes", "mellitus", "."])
        self.assertEqual(rows[2][3], "DISEASE[1]")
        self.assertEqual(rows[3][3], "DISEASE[1]")
        self.assertEqual(rows[4][3], "_")
        self.assertEqual(summary.snapped_token_boundaries, 0)

    def test_sentence_ids_advance_across_sentences(self) -> None:
        text = "No pneumonia today. Mother had lymphoma. Fever resolved."
        tsv = _artifact_to_webanno_tsv(text, [], note_id="TSV-N002", summary=SerializerSummary())
        ids = [row[0] for row in _token_rows(tsv)]

        self.assertIn("1-1", ids)
        self.assertIn("2-1", ids)
        self.assertIn("3-1", ids)

    def test_comma_after_entity_is_separate_token_and_span_survives(self) -> None:
        text = "Patient reports fever, chills absent."
        start = text.index("fever")
        end = start + len("fever")
        summary = SerializerSummary()

        tsv = _artifact_to_webanno_tsv(
            text,
            [
                {
                    "text": "fever",
                    "start_char": start,
                    "end_char": end,
                    "label": "DISEASE",
                }
            ],
            note_id="TSV-N003",
            summary=summary,
        )
        rows = _token_rows(tsv)
        fever_row = next(row for row in rows if row[2] == "fever")
        comma_row = rows[rows.index(fever_row) + 1]

        self.assertEqual(fever_row[3], "DISEASE")
        self.assertEqual(comma_row[2], r"\,")
        self.assertEqual(comma_row[3], "_")
        self.assertEqual(summary.snapped_token_boundaries, 0)

    def test_internal_whitespace_does_not_emit_space_token(self) -> None:
        text = "Patient has a PICC   line."
        start = text.index("PICC")
        end = text.index("line") + len("line")

        tsv = _artifact_to_webanno_tsv(
            text,
            [
                {
                    "text": "PICC   line",
                    "start_char": start,
                    "end_char": end,
                    "label": "PROCEDURE",
                }
            ],
            note_id="TSV-N004",
            summary=SerializerSummary(),
        )
        rows = _token_rows(tsv)

        self.assertNotIn("   ", [row[2] for row in rows])
        self.assertEqual(rows[3][2], "PICC")
        self.assertEqual(rows[3][3], "PROCEDURE[1]")
        self.assertEqual(rows[4][2], "line")
        self.assertEqual(rows[4][3], "PROCEDURE[1]")

    def test_non_aligned_span_snaps_outward_and_is_counted(self) -> None:
        text = "Patient has diabetes mellitus."
        start = text.index("diabetes") + 1
        end = text.index("mellitus") + len("mellitus") - 1
        summary = SerializerSummary()

        tsv = _artifact_to_webanno_tsv(
            text,
            [
                {
                    "text": text[start:end],
                    "start_char": start,
                    "end_char": end,
                    "label": "DISEASE",
                }
            ],
            note_id="TSV-N005",
            summary=summary,
        )
        rows = _token_rows(tsv)

        self.assertEqual(rows[2][2], "diabetes")
        self.assertEqual(rows[2][3], "DISEASE[1]")
        self.assertEqual(rows[3][2], "mellitus")
        self.assertEqual(rows[3][3], "DISEASE[1]")
        self.assertEqual(summary.snapped_token_boundaries, 1)


if __name__ == "__main__":
    unittest.main()
