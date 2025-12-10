# E:\adni_python\code\config.py
from pathlib import Path

# Raw DICOM roots on your external hard disk (H:)
RAW_ROOTS = [
    r"H:\adni\1st file",
    r"H:\adni\2nd file",
    r"H:\adni\3rd file",
    r"H:\adni\4th file",
    r"H:\adni\5th file",
    r"H:\adni\6th file",
    r"H:\adni\7th file",
    r"H:\adni\8th file",
    r"H:\adni\9th file",
    r"H:\adni\10th file",
]

# Project directories on E:
PROJECT_ROOT = Path(r"E:\adni_python")
CODE_DIR = PROJECT_ROOT / "code"
OUTPUT_RAW_NIFTI = PROJECT_ROOT / "outputs" / "raw_nifti"
OUTPUT_DERIV = PROJECT_ROOT / "outputs" / "derivatives"
LOG_DIR = PROJECT_ROOT / "logs"
ATLAS_DIR = PROJECT_ROOT / "atlas"

# Reference mask path (we'll add this later)
MNI_CEREB_MASK = ATLAS_DIR / "cereb_gm_mask_mni.nii.gz"