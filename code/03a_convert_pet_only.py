# E:\adni_python\code\03a_convert_pet_only.py
from pathlib import Path
import shutil
import dicom2nifti
import pandas as pd
from config import LOG_DIR, OUTPUT_RAW_NIFTI
from convert_pet_dynamic_safe import convert_pet_series_safe

def pet_out_path(subject, session):
    sub = subject.replace("_","")
    return OUTPUT_RAW_NIFTI / f"sub-{sub}" / f"ses-{session}" / "pet" / f"sub-{sub}_ses-{session}_tracer-FDG_pet.nii.gz"

def convert_pet_series(series_dir, out_nii):
    out_dir = out_nii.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp_convert"
    tmp.mkdir(exist_ok=True)
    try:
        # First try standard conversion
        dicom2nifti.convert_directory(series_dir, tmp, compression=True, reorient=True)
        niis = list(tmp.glob("*.nii*"))
        if not niis:
            raise RuntimeError("No NIfTI produced")
        shutil.move(str(niis), str(out_nii))
        return "standard"
    except Exception as e1:
        # Fallback: safe dynamic average
        try:
            convert_pet_series_safe(series_dir, out_nii)
            return "fallback"
        except Exception as e2:
            raise RuntimeError(f"standard+fallback failed: {e1} | {e2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    sel = pd.read_csv(LOG_DIR / "adni_selected_series_pet_only.csv")
    failures = []
    for _, row in sel.iterrows():
        series_dir = Path(row["series_dir"])
        out_nii = pet_out_path(row["Subject"], row["Session"])
        if out_nii.exists():
            print(f"Exists, skip: {out_nii}")
            continue
        print(f"Converting PET | {row['Subject']} {row['Session']}")
        try:
            mode = convert_pet_series(series_dir, out_nii)
            if mode == "fallback":
                print(f"  -> used fallback averaging for {series_dir}")
        except Exception as e:
            print(f"FAILED PET | {row['Subject']} {row['Session']} | {series_dir} | {e}")
            failures.append({
                "Subject": row["Subject"], "Session": row["Session"],
                "series_dir": str(series_dir), "error": str(e)
            })
    if failures:
        out_csv = LOG_DIR / "convert_failures_pet_only.csv"
        pd.DataFrame(failures).to_csv(out_csv, index=False)
        print(f"Wrote failures: {out_csv}")

if __name__ == "__main__":
    main()
