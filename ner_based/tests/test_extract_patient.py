from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.ner.extract_patient import (
    DEFAULT_NOTE_TITLES,
    DEFAULT_OUTPUT_PATH,
    LOG,
    PatientExtractionError,
    PatientNoteExtractionConfig,
    extract_patient_notes,
)
from src.ner.preannotate import load_notes


CLI_TEST_DEPS_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("typer", "rich")
)


class PatientExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.notes_path = self.root / "cohort_notes.parquet"
        self.cohort_path = self.root / "cohort_subset.csv"
        self.output_path = self.root / "extracted" / "patient.parquet"
        self.long_text = ("Complete clinical note. " * 320) + "FULL_TEXT_TAIL"

        self.notes = pd.DataFrame(
            {
                "PERSON_ID": [101, 101, 101, 101, 101, 101, 102, 103],
                "NOTE_ID": [30, 20, 10, 40, 50, 60, 70, 80],
                "NOTE_TITLE": [
                    "Progress Notes",
                    "Consults",
                    "Consults",
                    "Radiology",
                    "H&P",
                    "Discharge Summary",
                    "H&P",
                    "Radiology",
                ],
                "NOTE_TEXT": [
                    self.long_text,
                    "Second consultation.",
                    "First consultation.",
                    "Disallowed title.",
                    "   ",
                    None,
                    "Different patient.",
                    "No requested titles.",
                ],
                "NOTE_DATETIME": pd.to_datetime(
                    [
                        "2024-01-02 12:00",
                        "2024-01-01 09:00",
                        "2024-01-01 09:00",
                        "2024-01-03 09:00",
                        "2024-01-04 09:00",
                        "2024-01-05 09:00",
                        "2024-01-01 08:00",
                        "2024-01-01 08:00",
                    ]
                ),
                "VISIT_OCCURRENCE_ID": [
                    1003,
                    1002,
                    1001,
                    1004,
                    1005,
                    1006,
                    2001,
                    3001,
                ],
            }
        )
        self.notes.to_parquet(self.notes_path, index=False)
        pd.DataFrame(
            {
                "PERSON_ID": [101, 102, 103, 104],
                "LABEL": [1, 0, 0, 1],
            }
        ).to_csv(self.cohort_path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def config(
        self,
        person_id: object = "101",
        *,
        notes_path: Path | None = None,
        output_path: Path | None = None,
    ) -> PatientNoteExtractionConfig:
        return PatientNoteExtractionConfig(
            notes_parquet=notes_path or self.notes_path,
            cohort_csv=self.cohort_path,
            person_ids=(person_id,),
            output=output_path or self.output_path,
        )

    def test_extracts_filtered_full_notes_with_pushdown_and_counts(self) -> None:
        self.output_path.parent.mkdir(parents=True)
        self.output_path.write_text("old target")
        sibling = self.output_path.with_name("keep.txt")
        sibling.write_text("keep sibling")

        with patch(
            "src.ner.extract_patient.pd.read_parquet",
            wraps=pd.read_parquet,
        ) as read_parquet:
            with self.assertLogs(LOG.name, level="INFO") as captured:
                summary = extract_patient_notes(self.config())

        self.assertEqual(read_parquet.call_count, 1)
        read_kwargs = read_parquet.call_args.kwargs
        self.assertEqual(read_kwargs["filters"], [("PERSON_ID", "==", 101)])
        self.assertEqual(
            read_kwargs["columns"],
            [
                "NOTE_ID",
                "PERSON_ID",
                "NOTE_TEXT",
                "NOTE_TITLE",
                "NOTE_DATETIME",
                "VISIT_OCCURRENCE_ID",
            ],
        )

        extracted = pd.read_parquet(self.output_path)
        self.assertEqual(extracted["PERSON_ID"].tolist(), [101, 101, 101])
        self.assertEqual(extracted["NOTE_ID"].tolist(), [10, 20, 30])
        self.assertEqual(
            extracted["NOTE_TITLE"].tolist(),
            ["Consults", "Consults", "Progress Notes"],
        )
        self.assertTrue(extracted.loc[2, "NOTE_TEXT"].endswith("FULL_TEXT_TAIL"))
        self.assertGreater(len(extracted.loc[2, "NOTE_TEXT"]), 6000)
        self.assertEqual(extracted["LABEL"].tolist(), [1, 1, 1])
        self.assertEqual(
            list(extracted.columns),
            [
                "NOTE_ID",
                "PERSON_ID",
                "NOTE_TEXT",
                "NOTE_TITLE",
                "LABEL",
                "NOTE_DATETIME",
                "VISIT_OCCURRENCE_ID",
            ],
        )
        self.assertEqual(sibling.read_text(), "keep sibling")

        self.assertEqual(summary.total_notes, 3)
        self.assertEqual(
            summary.counts_by_title,
            {
                "Progress Notes": 1,
                "Consults": 2,
                "H&P": 0,
                "Discharge Summary": 0,
            },
        )
        self.assertEqual(
            [record.getMessage() for record in captured.records],
            [
                "Total notes extracted: 3",
                "Progress Notes: 1",
                "Consults: 2",
                "H&P: 0",
                "Discharge Summary: 0",
            ],
        )
        self.assertNotIn("FULL_TEXT_TAIL", "\n".join(captured.output))

        preannotation_notes = load_notes(self.output_path)
        self.assertEqual(
            [note.note_id for note in preannotation_notes],
            ["10", "20", "30"],
        )
        self.assertTrue(all(note.patient_id == "101" for note in preannotation_notes))
        self.assertTrue(preannotation_notes[2].text.endswith("FULL_TEXT_TAIL"))

    def test_patient_not_in_cohort_is_a_hard_error(self) -> None:
        with self.assertRaisesRegex(PatientExtractionError, "^patient not in cohort$"):
            extract_patient_notes(self.config("999"))
        self.assertFalse(self.output_path.exists())

    def test_cohort_patient_with_no_requested_titles_is_distinct_error(self) -> None:
        for person_id in ("103", "104"):
            with self.subTest(person_id=person_id):
                with self.assertRaisesRegex(
                    PatientExtractionError,
                    "^patient in cohort but no notes of requested titles$",
                ):
                    extract_patient_notes(self.config(person_id))
                self.assertFalse(self.output_path.exists())

    def test_incompatible_person_id_is_rejected_before_the_parquet_read(self) -> None:
        with patch("src.ner.extract_patient.pd.read_parquet") as read_parquet:
            with self.assertRaisesRegex(
                PatientExtractionError,
                "incompatible with parquet PERSON_ID type int64",
            ):
                extract_patient_notes(self.config("not-an-integer"))
        read_parquet.assert_not_called()

    def test_string_person_id_preserves_leading_zeroes(self) -> None:
        notes_path = self.root / "string_ids.parquet"
        cohort_path = self.root / "string_cohort.csv"
        output_path = self.root / "string_output.parquet"
        pd.DataFrame(
            {
                "PERSON_ID": ["00101", "00102"],
                "NOTE_ID": ["N2", "N1"],
                "NOTE_TITLE": ["Consults", "Consults"],
                "NOTE_TEXT": ["String patient.", "Other patient."],
            }
        ).to_parquet(notes_path, index=False)
        pd.DataFrame(
            {
                "PERSON_ID": ["00101", "00102"],
                "LABEL": [1, 0],
            }
        ).to_csv(cohort_path, index=False)

        config = PatientNoteExtractionConfig(
            notes_parquet=notes_path,
            cohort_csv=cohort_path,
            person_ids=("00101",),
            output=output_path,
        )
        extract_patient_notes(config)

        extracted = pd.read_parquet(output_path)
        self.assertEqual(extracted["PERSON_ID"].tolist(), ["00101"])
        self.assertEqual(extracted["NOTE_ID"].tolist(), ["N2"])
        self.assertNotIn("NOTE_DATETIME", extracted.columns)
        self.assertNotIn("VISIT_OCCURRENCE_ID", extracted.columns)

    def test_duplicate_note_ids_are_rejected(self) -> None:
        duplicate_path = self.root / "duplicate_notes.parquet"
        duplicate = pd.concat(
            [
                self.notes,
                pd.DataFrame(
                    {
                        "PERSON_ID": [101],
                        "NOTE_ID": [10],
                        "NOTE_TITLE": ["Consults"],
                        "NOTE_TEXT": ["Duplicate identifier."],
                        "NOTE_DATETIME": [pd.Timestamp("2024-01-06")],
                        "VISIT_OCCURRENCE_ID": [1007],
                    }
                ),
            ],
            ignore_index=True,
        )
        duplicate.to_parquet(duplicate_path, index=False)

        with self.assertRaisesRegex(
            PatientExtractionError,
            "^duplicate NOTE_ID values after filtering: 2 rows$",
        ):
            extract_patient_notes(self.config(notes_path=duplicate_path))
        self.assertFalse(self.output_path.exists())

    def test_existing_directory_output_is_not_modified(self) -> None:
        output_dir = self.root / "existing_output"
        output_dir.mkdir()
        child = output_dir / "keep.txt"
        child.write_text("keep")

        with self.assertRaisesRegex(
            PatientExtractionError,
            "^output path is an existing directory$",
        ):
            extract_patient_notes(self.config(output_path=output_dir))

        self.assertTrue(output_dir.is_dir())
        self.assertEqual(child.read_text(), "keep")


@unittest.skipUnless(
    CLI_TEST_DEPS_AVAILABLE,
    "Typer and Rich are required for CLI help validation",
)
class ExtractPatientCliHelpTest(unittest.TestCase):
    def test_help_documents_all_options(self) -> None:
        from typer.testing import CliRunner

        from src.cli import app

        result = CliRunner().invoke(
            app,
            ["extract-patient", "--help"],
            color=False,
            env={"COLUMNS": "240"},
        )

        self.assertEqual(result.exit_code, 0, result.output)
        for option in (
            "--notes-parquet",
            "--cohort-csv",
            "--person-id",
            "--note-titles",
            "--output",
        ):
            self.assertIn(option, result.output)
        self.assertIn(",".join(DEFAULT_NOTE_TITLES), result.output)
        self.assertIn(str(DEFAULT_OUTPUT_PATH), result.output)
        self.assertGreaterEqual(result.output.lower().count("required"), 3)

    def test_cli_success_output_contains_counts_only(self) -> None:
        from typer.testing import CliRunner

        from src.cli import app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes_path = root / "notes.parquet"
            cohort_path = root / "cohort.csv"
            output_path = root / "patient.parquet"
            pd.DataFrame(
                {
                    "PERSON_ID": [101],
                    "NOTE_ID": [1],
                    "NOTE_TITLE": ["Consults"],
                    "NOTE_TEXT": ["Never print this clinical text."],
                }
            ).to_parquet(notes_path, index=False)
            pd.DataFrame(
                {
                    "PERSON_ID": [101],
                    "LABEL": [1],
                }
            ).to_csv(cohort_path, index=False)

            result = CliRunner().invoke(
                app,
                [
                    "extract-patient",
                    "--notes-parquet",
                    str(notes_path),
                    "--cohort-csv",
                    str(cohort_path),
                    "--person-id",
                    "101",
                    "--output",
                    str(output_path),
                ],
                color=False,
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Total notes extracted: 1", result.output)
        self.assertIn("Consults: 1", result.output)
        self.assertNotIn("Never print this clinical text.", result.output)
        self.assertNotIn("Seed", result.output)
        self.assertNotIn("Run dir", result.output)
        self.assertNotIn(str(notes_path), result.output)


if __name__ == "__main__":
    unittest.main()
