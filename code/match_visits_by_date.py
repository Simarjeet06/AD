# E:\adni_python\code\match_visits_by_date.py
import pandas as pd
import numpy as np
from pathlib import Path
from data_config import RAW_NIFTI, MASTER_OUT, MATCHED_OUT

def parse_visdate(s):
    if pd.isna(s): return pd.NaT
    # Try common ADNI formats
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

def parse_session_folder(token):
    # If token is YYYYMMDD -> parse to date, else NaT
    s = str(token)
    if len(s) == 8 and s.isdigit():
        try:
            return pd.to_datetime(s, format="%Y%m%d")
        except Exception:
            return pd.NaT
    return pd.NaT

def ptid_from_sub(sub):
    s = str(sub)
    if "_" in s: return s
    if len(s) == 8 and s[3] == "S":
        return f"{s[:3]}_S_{s[4:]}"
    if len(s) == 8:
        return f"{s[:3]}_S_{s[3:]}"
    return s

def build_session_index():
    rows = []
    for subdir in RAW_NIFTI.glob("sub-*"):
        sub_token = subdir.name.replace("sub-","")
        ptid = ptid_from_sub(sub_token)
        for sesdir in subdir.glob("ses-*"):
            ses_token = sesdir.name.replace("ses-","")
            ses_date = parse_session_folder(ses_token)
            anat = list((sesdir/"anat").glob("*_T1w.nii.gz"))
            pet = list((sesdir/"pet").glob("*.nii.gz"))
            rows.append({
                "PTID": ptid,
                "session_token": ses_token,
                "session_date": ses_date,
                "anat_path": str(anat[0]) if anat else "",
                "pet_path": str(pet) if pet else "",
                "has_t1_match": bool(anat),
                "has_pet_match": bool(pet),
            })
    return pd.DataFrame(rows)

def nearest_session(visit_date, sess_df, max_days=90):
    if pd.isna(visit_date) or sess_df.empty:
        return None
    if sess_df["session_date"].isna().all():
        return None
    diffs = (sess_df["session_date"] - visit_date).abs()
    idx = diffs.idxmin()
    if pd.isna(sess_df.loc[idx, "session_date"]):
        return None
    return sess_df.loc[idx].to_dict() if diffs.loc[idx].days <= max_days else None

def coalesce_visdate(row):
    # choose first non-null date among available columns
    for c in ["VISDATE","VISDATE_cdr","VISDATE_dem","EXAMDATE","EXAMDATE_dx","EXAMDATE_blc"]:
        if c in row and pd.notna(row[c]):
            return row[c]
    return pd.NaT

def main():
    master = pd.read_csv(MASTER_OUT, low_memory=False)

    # Build a VISDATE_ANY column from whichever date columns exist
    for c in ["VISDATE","VISDATE_cdr","VISDATE_dem","EXAMDATE","EXAMDATE_dx","EXAMDATE_blc"]:
        if c in master.columns:
            master[c] = master[c].apply(parse_visdate)
    master["VISDATE_ANY"] = master.apply(coalesce_visdate, axis=1)

    sess = build_session_index()
    sess_by_ptid = {ptid: g.reset_index(drop=True) for ptid, g in sess.groupby("PTID")}

    # Initialize new columns
    master["matched_session_token"] = ""
    master["matched_session_date"] = pd.NaT
    master["anat_path"] = ""
    master["pet_path"] = ""
    master["has_t1_match"] = False
    master["has_pet_match"] = False

    for i, row in master.iterrows():
        ptid = row.get("subject_id")
        vdate = row.get("VISDATE_ANY")
        if pd.isna(ptid) or pd.isna(vdate):
            continue
        g = sess_by_ptid.get(ptid)
        if g is None or g.empty:
            continue
        m = nearest_session(vdate, g, max_days=90)  # adjust window as needed
        if m is None:
            continue
        master.at[i,"matched_session_token"] = m["session_token"]
        master.at[i,"matched_session_date"] = m["session_date"]
        master.at[i,"anat_path"] = m["anat_path"]
        master.at[i,"pet_path"] = m["pet_path"]
        master.at[i,"has_t1_match"] = m["has_t1_match"]
        master.at[i,"has_pet_match"] = m["has_pet_match"]

    master.to_csv(MATCHED_OUT, index=False)
    print("Wrote:", MATCHED_OUT, "| rows:", len(master))
    print("Matched (any imaging):", int(((master.has_t1_match) | (master.has_pet_match)).sum()))
    print("Matched (t1 only):", int((master.has_t1_match & ~master.has_pet_match).sum()))
    print("Matched (pet only):", int((~master.has_t1_match & master.has_pet_match).sum()))
    print("Matched (t1+pet):", int((master.has_t1_match & master.has_pet_match).sum()))

if __name__ == "__main__":
    main()
