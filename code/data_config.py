# E:\adni_python\code\data_config.py
from pathlib import Path

PROJECT = Path(r"E:\adni_python")
RELATION = PROJECT / "relational_data"
OUT = PROJECT / "outputs"

# All files are CSV (per your folder listing)
ADAS_FILE = RELATION / "All_Subjects_ADAS_21Aug2025.csv"
CDR_FILE = RELATION / "All_Subjects_CDR_21Aug2025.csv"
MMSE_FILE = RELATION / "All_Subjects_MMSE_21Aug2025.csv"
PTDEMOG_FILE = RELATION / "All_Subjects_PTDEMOG_21Aug2025.csv"
APOERS_FILE = RELATION / "All_Subjects_APOERES_21Aug2025.csv"
BLCHANGE_FILE = RELATION / "All_Subjects_BLCHANGE_21Aug2025.csv"
DXSUM_FILE = RELATION / "All_Subjects_DXSUM_21Aug2025.csv"
KEY_MRI_FILE = RELATION / "All_Subjects_Key_MRI_21Aug2025.csv"
KEY_PET_FILE = RELATION / "All_Subjects_Key_PET_21Aug2025.csv"

RAW_NIFTI = PROJECT / "outputs" / "raw_nifti"
DERIV = PROJECT / "outputs" / "derivatives"

MASTER_OUT = OUT / "master_longitudinal.csv"
MATCHED_OUT = OUT / "master_with_imaging_match.csv"
