# E:\adni_python\code\01_index_series.py
import os
from pathlib import Path
import pandas as pd
import pydicom
from .config import RAW_ROOTS, LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

WANTED_TAGS = [
    "PatientID","StudyDate","StudyTime","SeriesDescription","SeriesNumber","Modality",
    "SeriesInstanceUID","Manufacturer","MagneticFieldStrength","ImageType","Rows","Columns","NumberOfFrames"
]

def read_header_fast(dcm_path):
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
    rec = {}
    for t in WANTED_TAGS:
        val = getattr(ds, t, None)
        if t == "ImageType" and val is not None:
            if isinstance(val, (list, tuple)):
                val = "\\".join([str(x) for x in val])
        rec[t] = "" if val is None else str(val)
    try:
        ri = ds.RadiopharmaceuticalInformationSequence[0]
        rec["Radiopharm"] = getattr(ri, "Radiopharmaceutical", "")
    except Exception:
        rec["Radiopharm"] = ""
    return rec

def index_series(root):
    rows = []
    for dirpath, _, filenames in os.walk(root):
        dcm_files = [f for f in filenames if f.lower().endswith(".dcm")]
        if len(dcm_files) < 5:
            continue
        sample = Path(dirpath) / dcm_files[0]
        try:
            hdr = read_header_fast(sample)
        except Exception:
            continue
        hdr["series_dir"] = dirpath
        hdr["n_files"] = len(dcm_files)
        rows.append(hdr)
    return pd.DataFrame(rows)

def main():
    dfs = []
    for r in RAW_ROOTS:
        if not Path(r).exists():
            print(f"Warning: missing root {r}")
            continue
        print(f"Indexing {r} ...")
        dfs.append(index_series(r))
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df["Subject"] = df["series_dir"].str.extract(r"(ADNI[\\/](\d{3}_S_\d{4}))")[1]
    df["Session"] = df["StudyDate"]
    out_csv = LOG_DIR / "adni_series_index.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")

if __name__ == "__main__":
    main()