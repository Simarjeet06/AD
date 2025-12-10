# E:\adni_python\code\03_convert_to_nifti.py
from pathlib import Path
import shutil
import dicom2nifti
import pandas as pd
from config import LOG_DIR, OUTPUT_RAW_NIFTI

def convert_series(series_dir, out_nii):
    out_dir = out_nii.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp_convert"
    tmp.mkdir(exist_ok=True)
    try:
        # Try a normal conversion (handles both 3D and 4D if consistent)
        dicom2nifti.convert_directory(series_dir, tmp, compression=True, reorient=True)
        niis = list(tmp.glob("*.nii*"))
        if not niis:
            raise RuntimeError("No NIfTI produced")
        shutil.move(str(niis[0]), str(out_nii))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def out_path(subject, session, which):
    sub = subject.replace("_","")
    if which == "T1":
        return OUTPUT_RAW_NIFTI / f"sub-{sub}" / f"ses-{session}" / "anat" / f"sub-{sub}_ses-{session}_T1w.nii.gz"
    else:
        return OUTPUT_RAW_NIFTI / f"sub-{sub}" / f"ses-{session}" / "pet" / f"sub-{sub}_ses-{session}_tracer-FDG_pet.nii.gz"

def main():
    sel = pd.read_csv(LOG_DIR / "adni_selected_series.csv")
    failures = []
    for _, row in sel.iterrows():
        series_dir = Path(row["series_dir"])
        out_nii = out_path(row["Subject"], row["Session"], row["which"])
        if out_nii.exists():
            print(f"Exists, skip: {out_nii}")
            continue
        print(f"Converting {row['which']} | {row['Subject']} {row['Session']}")
        try:
            convert_series(series_dir, out_nii)
        except Exception as e:
            # DO NOT RAISE — record and continue
            print(f"FAILED: {row['which']} | {row['Subject']} {row['Session']} | {series_dir} | {e}")
            failures.append({
                "Subject": row["Subject"],
                "Session": row["Session"],
                "which": row["which"],
                "series_dir": str(series_dir),
                "error": str(e)
            })
            continue

    if failures:
        fail_csv = LOG_DIR / "convert_failures.csv"
        pd.DataFrame(failures).to_csv(fail_csv, index=False)
        print(f"Wrote failures log: {fail_csv}")

if __name__ == "__main__":
    main()

