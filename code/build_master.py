# E:\adni_python\code\build_master.py
import sys
from pathlib import Path

# Ensure this script folder is on sys.path, then import config
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_config import *  # brings ADAS_FILE, etc.

import pandas as pd
import numpy as np

def read_xl(path, usecols=None):
    # Auto-choose CSV vs Excel by extension
    p = str(path).lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path, engine="openpyxl", usecols=usecols)
    return pd.read_csv(path)

def norm_subject_id(x):
    if pd.isna(x): return np.nan
    return str(x).strip()

def norm_visit_code(v):
    if pd.isna(v): return np.nan
    return str(v).strip().lower().replace(" ", "")

def keep_latest_duplicates(df, subset_cols, date_col=None):
    if date_col and date_col in df.columns:
        df2 = df.sort_values(by=[*subset_cols, date_col])
        return df2.groupby(subset_cols, as_index=False).tail(1)
    return df.drop_duplicates(subset=subset_cols, keep="last")

def load_adas():
    df = read_xl(ADAS_FILE)
    cols = ["PHASE","PTID","RID","VISCODE","VISCODE2","VISDATE","TOTSCORE","TOTAL13"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["subject_id"] = df["PTID"].map(norm_subject_id)
    df["visit"] = df["VISCODE2"].fillna(df["VISCODE"]).map(norm_visit_code)
    df = keep_latest_duplicates(df, ["subject_id","visit"], date_col="VISDATE" if "VISDATE" in df.columns else None)
    df.rename(columns={"TOTSCORE":"ADAS_TOTSCORE","TOTAL13":"ADAS_TOTAL13"}, inplace=True)
    return df[["subject_id","visit","PHASE","RID","ADAS_TOTSCORE","ADAS_TOTAL13","VISDATE"]]

def load_cdr():
    df = read_xl(CDR_FILE)
    cols = ["PHASE","PTID","RID","VISCODE","VISCODE2","VISDATE","CDGLOBAL","CDRSB"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["subject_id"] = df["PTID"].map(norm_subject_id)
    df["visit"] = df["VISCODE2"].fillna(df["VISCODE"]).map(norm_visit_code)
    df = keep_latest_duplicates(df, ["subject_id","visit"], date_col="VISDATE" if "VISDATE" in df.columns else None)
    df.rename(columns={"CDGLOBAL":"CDR_GLOBAL","CDRSB":"CDR_SOB"}, inplace=True)
    return df[["subject_id","visit","PHASE","RID","CDR_GLOBAL","CDR_SOB","VISDATE"]]

def load_mmse():
    df = read_xl(MMSE_FILE)
    cols = ["PHASE","PTID","RID","VISCODE","VISCODE2","VISDATE","MMSCORE"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["subject_id"] = df["PTID"].map(norm_subject_id)
    df["visit"] = df["VISCODE2"].fillna(df["VISCODE"]).map(norm_visit_code)
    df = keep_latest_duplicates(df, ["subject_id","visit"], date_col="VISDATE" if "VISDATE" in df.columns else None)
    df.rename(columns={"MMSCORE":"MMSE_SCORE"}, inplace=True)
    return df[["subject_id","visit","PHASE","RID","MMSE_SCORE","VISDATE"]]

def load_demog():
    df = read_xl(PTDEMOG_FILE)
    cols = ["PHASE","PTID","RID","VISCODE","VISCODE2","VISDATE","PTGENDER","PTDOB","PTDOBYY","PTEDUCAT"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["subject_id"] = df["PTID"].map(norm_subject_id)
    df["visit"] = df["VISCODE2"].fillna(df["VISCODE"]).map(norm_visit_code)
    df = keep_latest_duplicates(df, ["subject_id","visit"], date_col="VISDATE" if "VISDATE" in df.columns else None)
    return df[["subject_id","visit","PHASE","RID","PTGENDER","PTDOB","PTDOBYY","PTEDUCAT","VISDATE"]]

def load_apoe():
    df = read_xl(APOERS_FILE)
    cols_possible = {"PTID","RID","PHASE","VISCODE","VISCODE2","APTESTDT","GENOTYPE"}
    cols = [c for c in df.columns if c in cols_possible]
    df = df[cols].copy()
    df["subject_id"] = df["PTID"].map(norm_subject_id) if "PTID" in df else np.nan
    if "VISCODE2" in df:
        df["visit"] = df["VISCODE2"].fillna(df.get("VISCODE")).map(norm_visit_code)
    elif "VISCODE" in df:
        df["visit"] = df["VISCODE"].map(norm_visit_code)
    else:
        df["visit"] = "bl"
    df.rename(columns={"GENOTYPE":"APOE_GENOTYPE"}, inplace=True)
    df = keep_latest_duplicates(df, ["subject_id","visit"], date_col="APTESTDT" if "APTESTDT" in df.columns else None)
    return df[["subject_id","visit","PHASE","RID","APOE_GENOTYPE"]]

def load_blchange():
    df = read_xl(BLCHANGE_FILE)
    cols = ["PHASE","PTID","RID","VISCODE","VISCODE2","EXAMDATE","BCADAS","BCMMSE","BCCDR"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["subject_id"] = df["PTID"].map(norm_subject_id)
    df["visit"] = df["VISCODE2"].fillna(df["VISCODE"]).map(norm_visit_code)
    df = keep_latest_duplicates(df, ["subject_id","visit"], date_col="EXAMDATE" if "EXAMDATE" in df.columns else None)
    return df[["subject_id","visit","PHASE","RID","BCADAS","BCMMSE","BCCDR","EXAMDATE"]]

def load_dxsum():
    df = read_xl(DXSUM_FILE)
    cols = ["PHASE","PTID","RID","VISCODE","VISCODE2","EXAMDATE","DIAGNOSIS","DXNORM","DXMCI","DXAD"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["subject_id"] = df["PTID"].map(norm_subject_id)
    df["visit"] = df["VISCODE2"].fillna(df["VISCODE"]).map(norm_visit_code)
    df = keep_latest_duplicates(df, ["subject_id","visit"], date_col="EXAMDATE" if "EXAMDATE" in df.columns else None)
    return df[["subject_id","visit","PHASE","RID","DIAGNOSIS","DXNORM","DXMCI","DXAD","EXAMDATE"]]

def load_key_mri():
    df = read_xl(KEY_MRI_FILE)
    cols = ["image_id","subject_id","image_visit","image_date","series_type","loni_image"]
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()
    df["subject_id"] = df["subject_id"].map(norm_subject_id)
    df["visit"] = df["image_visit"].map(norm_visit_code)
    df.rename(columns={"image_date":"image_date_mri","loni_image":"loni_image_mri"}, inplace=True)
    df = keep_latest_duplicates(df, ["subject_id","visit"])
    return df[["subject_id","visit","image_id","image_date_mri","series_type","loni_image_mri"]]

def load_key_pet():
    df = read_xl(KEY_PET_FILE)
    cols = ["image_id","subject_id","image_visit","image_date","radiopharmaceutical","pet_description"]
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()
    df["subject_id"] = df["subject_id"].map(norm_subject_id)
    df["visit"] = df["image_visit"].map(norm_visit_code)
    df.rename(columns={"image_date":"image_date_pet"}, inplace=True)
    df = keep_latest_duplicates(df, ["subject_id","visit"])
    return df[["subject_id","visit","image_id","image_date_pet","radiopharmaceutical","pet_description"]]

def scan_imaging_availability():
    rows = []
    for subdir in RAW_NIFTI.glob("sub-*"):
        for sesdir in subdir.glob("ses-*"):
            sub = subdir.name.replace("sub-","")
            ses = sesdir.name.replace("ses-","")
            has_t1 = any((sesdir/"anat").glob("*_T1w.nii.gz"))
            has_pet = any((sesdir/"pet").glob("*.nii.gz"))
            rows.append({
                "subject_id": sub_with_underscores(sub),
                "visit": ses_to_viscode(ses),
                "has_t1": has_t1,
                "has_pet": has_pet
            })
    return pd.DataFrame(rows)

def sub_with_underscores(sub):
    if "_" in sub: return sub
    if len(sub) == 8 and sub[3] == "S":
        return f"{sub[:3]}_S_{sub[4:]}"
    if len(sub) == 8:
        return f"{sub[:3]}_S_{sub[3:]}"
    return sub

def ses_to_viscode(ses):
    # Session token is often YYYYMMDD; keep it as-is (we'll date-match later)
    return str(ses).lower().strip()

def main():
    adas = load_adas()
    cdr = load_cdr()
    mmse = load_mmse()
    dem = load_demog()
    apoe = load_apoe()
    blc = load_blchange()
    dx = load_dxsum()
    kmri = load_key_mri()
    kpet = load_key_pet()

    master = adas.merge(cdr, on=["subject_id","visit","PHASE","RID"], how="outer", suffixes=("","_cdr"))
    master = master.merge(mmse, on=["subject_id","visit","PHASE","RID"], how="outer")
    master = master.merge(dem, on=["subject_id","visit","PHASE","RID"], how="outer", suffixes=("","_dem"))
    master = master.merge(apoe, on=["subject_id","visit","PHASE","RID"], how="left")
    master = master.merge(blc, on=["subject_id","visit","PHASE","RID"], how="left", suffixes=("","_blc"))
    master = master.merge(dx, on=["subject_id","visit","PHASE","RID"], how="left", suffixes=("","_dx"))

    master = master.merge(kmri, on=["subject_id","visit"], how="left")
    master = master.merge(kpet, on=["subject_id","visit"], how="left", suffixes=("","_petkey"))

    avail = scan_imaging_availability()
    master = master.merge(avail, on=["subject_id","visit"], how="left")

    master.sort_values(by=["subject_id","visit"], inplace=True)
    for col in ["PTGENDER","PTEDUCAT","PTDOB","PTDOBYY","APOE_GENOTYPE"]:
        if col in master.columns:
            master[col] = master.groupby("subject_id")[col].ffill().bfill()

    cols_first = [
        "subject_id","visit","PHASE","RID","has_t1","has_pet",
        "ADAS_TOTSCORE","ADAS_TOTAL13","CDR_GLOBAL","CDR_SOB","MMSE_SCORE",
        "PTGENDER","PTEDUCAT","PTDOB","PTDOBYY","APOE_GENOTYPE",
        "DIAGNOSIS","DXNORM","DXMCI","DXAD"
    ]
    cols_out = [c for c in cols_first if c in master.columns] + [c for c in master.columns if c not in cols_first]
    master = master[cols_out]

    OUT.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_OUT, index=False)
    print("Wrote:", MASTER_OUT, "| rows:", len(master))

if __name__ == "__main__":
    print("ADAS_FILE:", ADAS_FILE)
    main()
