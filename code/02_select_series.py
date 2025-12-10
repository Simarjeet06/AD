# E:\adni_python\code\02_select_series.py
import pandas as pd
from config import LOG_DIR

def to_int(x):
    try:
        return int(str(x))
    except Exception:
        return 0

def pick_t1(df_sub):
    # Filter structural T1 candidates
    cand = df_sub[
        (df_sub.Modality == "MR")
        & df_sub.SeriesDescription.str.contains("MPRAGE|MP-?RAGE|IR-FSPGR|T1", case=False, na=False)
    ]
    if cand.empty:
        return None
    cand = cand.copy()
    cand["is_corr"] = cand.SeriesDescription.str.contains("grad|gw|b1|n3|bias|corr", case=False, na=False)
    cand["Rows_i"] = cand["Rows"].apply(to_int)
    cand["Cols_i"] = cand["Columns"].apply(to_int)
    cand["pix"] = cand["Rows_i"] * cand["Cols_i"]
    cand["SeriesNumber_i"] = cand["SeriesNumber"].apply(to_int)
    cand = cand.sort_values(
        by=["is_corr", "pix", "n_files", "SeriesNumber_i"],
        ascending=[False, False, False, False],
    )
    # Return a single row (Series)
    return cand.iloc[0]

def pick_pet_fdg(df_sub):
    fdg_mask = (
        df_sub.SeriesDescription.str.contains("FDG", case=False, na=False)
        | df_sub.get("Radiopharm", "").astype(str).str.contains("FDG", case=False, na=False)
    )
    cand = df_sub[(df_sub.Modality == "PT") & fdg_mask]
    if cand.empty:
        return None
    cand = cand.copy()
    cand["is_avg"] = cand.SeriesDescription.str.contains("averag|avg|co-?reg", case=False, na=False)
    cand["SeriesNumber_i"] = cand["SeriesNumber"].apply(to_int)
    cand = cand.sort_values(by=["is_avg", "n_files", "SeriesNumber_i"], ascending=[False, False, False])
    # Return a single row (Series)
    return cand.iloc[0]

def main():
    df = pd.read_csv(LOG_DIR / "adni_series_index.csv")

    selected_rows = []
    # Loop per Subject + Session (StudyDate)
    for (sub, ses), dfg in df.groupby(["Subject", "Session"], dropna=True):
        t1_row = pick_t1(dfg)
        pet_row = pick_pet_fdg(dfg)

        if t1_row is not None:
            r = t1_row.to_dict()
            r["which"] = "T1"
            selected_rows.append(r)

        if pet_row is not None:
            r = pet_row.to_dict()
            r["which"] = "PET"
            selected_rows.append(r)

    sel = pd.DataFrame(selected_rows)
    out_csv = LOG_DIR / "adni_selected_series.csv"
    sel.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(sel)} selections")

if __name__ == "__main__":
    main()
