from pathlib import Path
import pandas as pd
from src.ner.extract_patient import PatientNoteExtractionConfig, extract_patient_notes

BASE   = Path('/sc/arion/projects/MRSA-HPI-MS/airms-app-host-and-hospital-adaptation-of-mrsa/mrsa_nlp/rule_based/data/interim/airms')
NOTES  = BASE / 'notes/all/cohort_notes.parquet'
COHORT = BASE / 'cohort_subset.csv'
OUT    = Path('/sc/arion/work/rademt02/airms_notes/extracted/cohort_all_notes.parquet')

cohort = pd.read_csv(COHORT).drop_duplicates('PERSON_ID')
ids = tuple(cohort.PERSON_ID.tolist())
print('cohort patients:', len(ids), '| label balance:', cohort.LABEL.value_counts().to_dict())

s = extract_patient_notes(PatientNoteExtractionConfig(
    notes_parquet=NOTES, cohort_csv=COHORT, person_ids=ids, output=OUT,
))
print('total notes:', s.total_notes)          # expect 20951
print('by title:', s.counts_by_title)
