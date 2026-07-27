from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ner.preannotate import (
    Note,
    parse_model_json,
    run_preannotation,
    synthetic_canned_responses,
    synthetic_fixture_notes,
    verify_model_response,
)


class PreannotationVerificationTest(unittest.TestCase):
    def test_verifies_required_synthetic_edge_cases(self) -> None:
        notes = {note.note_id: note for note in synthetic_fixture_notes()}
        responses = synthetic_canned_responses()

        first = verify_model_response(notes["SYN-N001"], responses["SYN-N001"])
        second = verify_model_response(notes["SYN-N002"], responses["SYN-N002"])

        self.assertEqual(first.stats.dropped_not_in_text, 1)
        self.assertEqual(first.stats.ambiguous, 1)
        self.assertEqual(first.stats.overlapping, 1)
        self.assertTrue(any(span["text"] == "pneumonia" for span in first.spans))
        self.assertTrue(any(span["text"] == "lymphoma" for span in first.spans))
        self.assertTrue(all("assertion" not in span for span in first.spans))
        self.assertTrue(all("temporality" not in span for span in first.spans))
        self.assertTrue(all("experiencer" not in span for span in first.spans))

        picc = next(span for span in second.spans if span["text"] == "PICC   line")
        self.assertEqual(notes["SYN-N002"].text[picc["start_char"] : picc["end_char"]], "PICC   line")

    def test_parse_model_json_strips_fences_and_substrings(self) -> None:
        fenced = """```json
{"entities": [{"text": "pneumonia", "label": "DISEASE"}]}
```"""
        self.assertEqual(parse_model_json(fenced)["entities"][0]["text"], "pneumonia")

        surrounded = 'Here is JSON: {"entities": []} done.'
        self.assertEqual(parse_model_json(surrounded), {"entities": []})

        failed = parse_model_json("not json")
        self.assertTrue(failed["parse_failed"])
        self.assertIn("raw_response", failed)

    def test_run_preannotation_writes_one_json_per_note(self) -> None:
        note = Note(
            note_id="UNIT-N001",
            patient_id="UNIT-P001",
            text="No pneumonia. fever fever.",
        )
        response = {
            "entities": [
                {
                    "text": "pneumonia",
                    "label": "DISEASE",
                },
                {
                    "text": "fever",
                    "label": "DISEASE",
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            totals = run_preannotation(
                notes=[note],
                out_dir=out_dir,
                canned_responses={"UNIT-N001": response},
            )
            artifact = json.loads((out_dir / "verified" / "UNIT-N001.json").read_text())

        self.assertEqual(totals.proposed, 2)
        self.assertEqual(totals.ambiguous, 1)
        self.assertEqual(totals.verified, 3)
        self.assertTrue(totals.counters_reconcile())
        self.assertEqual(artifact["note_id"], "UNIT-N001")
        self.assertEqual(len(artifact["spans"]), 3)
        self.assertEqual(
            set(artifact["spans"][0]),
            {
                "text",
                "start_char",
                "end_char",
                "label",
                "source",
                "ambiguous_occurrence",
                "occurrence_index",
                "overlaps_previous",
            },
        )

    def test_synthetic_totals_counter_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            totals = run_preannotation(
                notes=synthetic_fixture_notes(),
                out_dir=Path(tmp),
                canned_responses=synthetic_canned_responses(),
                overwrite=True,
            )

        self.assertEqual(totals.dropped_not_in_text, 1)
        self.assertEqual(totals.ambiguous, 1)
        self.assertEqual(totals.overlapping, 1)
        self.assertTrue(totals.counters_reconcile())


if __name__ == "__main__":
    unittest.main()
