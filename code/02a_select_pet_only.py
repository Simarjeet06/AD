# E:\adni_python\code\02a_select_pet_only.py
import pandas as pd
from config import LOG_DIR

def to_int(x):
    try:
        return int(str(x))
    except Exception:
        return 0

def pick_pet_fdg(df_sub):
    desc = df_sub.SeriesDescription.fillna("").str.lower()

    # PET FDG candidates
    fdg_mask = desc.str.contains("fdg") | df_sub.get("Radiopharm","").astype(str).str.lower().str.contains("fdg")
    cand = df_sub[(df_sub.Modality == "PT") & fdg_mask]
    if cand.empty:
        return None

    cand = cand.copy()
    cdesc = cand.SeriesDescription.fillna("").str.lower()

    # Strong positive indicators of static, analysis-ready PET
    good_static = (
        cdesc.str.contains("co[- ]?reg|co[_ ]?registered|coreg|registered|rigid") |
        cdesc.str.contains("avg|averag|mean|sum|static") |
        cdesc.str.contains("suvr|quant|quantified|std|standardized")
    )

    # Demote raw/dynamic/ctac-only
    bad_raw = (
        cdesc.str.contains("raw") |
        cdesc.str.contains(r"\bctac\b") |  # CT attenuation correction tag in desc
        cdesc.str.contains(r"\bac\b") |
        cdesc.str.contains("dynamic|frame")
    )

    # Score and sort
    cand["rank"] = 0
    cand.loc[good_static, "rank"] += 2
    cand.loc[bad_raw, "rank"] -= 2
    cand["SeriesNumber_i"] = cand["SeriesNumber"].apply(to_int)

    # More files often means better reconstructed stack for static series
    cand = cand.sort_values(by=["rank","n_files","SeriesNumber_i"], ascending=[False,False,False])

    return cand.iloc[0]

def main():
    df = pd.read_csv(LOG_DIR / "adni_series_index.csv")

    selected = []
    for (sub, ses), dfg in df.groupby(["Subject","Session"], dropna=True):
        pet = pick_pet_fdg(dfg)
        if pet is not None:
            r = pet.to_dict()
            r["which"] = "PET"
            selected.append(r)

    sel = pd.DataFrame(selected)
    out_csv = LOG_DIR / "adni_selected_series_pet_only.csv"
    sel.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(sel)} PET selections")

if __name__ == "__main__":
    main()