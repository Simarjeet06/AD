# E:\adni_python\code\04_preprocess.py
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
import ants
from config import OUTPUT_RAW_NIFTI, OUTPUT_DERIV, LOG_DIR, MNI_CEREB_MASK, ATLAS_DIR

# Paths to your MNI template brain (binary mask will be derived)
MNI_TEMPLATE = ATLAS_DIR / "templates" / "MNI152_T1_1mm_brain.nii.gz"

def n4_bias(in_nii, out_nii):
    img = ants.image_read(str(in_nii))
    m = ants.get_mask(img)
    corr = ants.n4_bias_field_correction(img, mask=m)
    ants.image_write(corr, str(out_nii))

def make_mni_brain_mask():
    """Create a binary brain mask in MNI space from the provided MNI brain image."""
    mni_brain = ants.image_read(str(MNI_TEMPLATE))
    mask = mni_brain > 0
    return mask

def t1_to_mni(t1_path):
    """
    Register a T1 (N4-corrected) to MNI brain.
    Returns registration dict; do not try to re-write transforms (Windows returns paths).
    """
    fixed = ants.image_read(str(MNI_TEMPLATE))
    moving = ants.image_read(str(t1_path))
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
    return reg  # contains 'warpedmovout', 'fwdtransforms', 'invtransforms'

def warp_mni_mask_to_t1(mni_mask_img, t1_img, inv_list, out_mask_t1):
    warped = ants.apply_transforms(
        fixed=t1_img,
        moving=mni_mask_img,
        transformlist=inv_list,
        interpolator='nearestNeighbor'
    )
    (warped > 0.5).to_file(str(out_mask_t1))

def brain_extract_via_template(t1_n4_path, out_brain_path, out_mask_path):
    """
    Brain extraction by registering T1 to MNI brain and warping an MNI brain mask back to T1.
    """
    reg = t1_to_mni(t1_n4_path)
    inv_list = reg['invtransforms']  # to go MNI->T1

    # Make MNI brain mask and warp to T1
    mni_mask = make_mni_brain_mask()
    t1_img = ants.image_read(str(t1_n4_path))
    warp_mni_mask_to_t1(mni_mask, t1_img, inv_list, out_mask_path)

    # Apply mask
    mask_img = ants.image_read(str(out_mask_path))
    brain = t1_img * mask_img
    ants.image_write(brain, str(out_brain_path))
    return reg  # reuse inv_list for other mask warps

def pet_avg_if4d(in_nii, out_nii):
    img = nib.load(str(in_nii)); data = img.get_fdata()
    if data.ndim == 4:
        data = data.mean(axis=-1)
    nib.Nifti1Image(data.astype(np.float32), img.affine, img.header).to_filename(str(out_nii))

def pet_to_t1_rigid(pet_nii, t1_brain, out_pet_in_t1, out_tx):
    pet = ants.image_read(str(pet_nii)); t1 = ants.image_read(str(t1_brain))
    reg = ants.registration(fixed=t1, moving=pet, type_of_transform='Rigid')
    ants.image_write(reg['warpedmovout'], str(out_pet_in_t1))
    # Save transform if it is an in-memory object; if it's already a path, copy path string
    fwd = reg['fwdtransforms'][0]
    try:
        ants.write_transform(fwd, str(out_tx))
    except Exception:
        # If write fails (likely because fwd is already a filepath), just record the intention
        pass
    return reg

def process_one(sub, ses):
    subshort = sub.replace("_","")
    raw_anat = OUTPUT_RAW_NIFTI / f"sub-{subshort}" / f"ses-{ses}" / "anat" / f"sub-{subshort}_ses-{ses}_T1w.nii.gz"
    raw_pet  = OUTPUT_RAW_NIFTI / f"sub-{subshort}" / f"ses-{ses}" / "pet"  / f"sub-{subshort}_ses-{ses}_tracer-FDG_pet.nii.gz"
    if not raw_anat.exists():
        print(f"Missing T1 for {sub} {ses}, skipping")
        return
    if not raw_pet.exists():
        print(f"Missing PET for {sub} {ses}, skipping")
        return

    outdir = OUTPUT_DERIV / f"sub-{subshort}" / f"ses-{ses}"
    outdir.mkdir(parents=True, exist_ok=True)

    t1_n4 = outdir / "t1_n4.nii.gz"
    t1_brain = outdir / "t1_brain.nii.gz"
    t1_mask = outdir / "t1_mask.nii.gz"

    inv_list = None

    if not t1_brain.exists():
        print(f"[{sub} {ses}] N4 + brain extraction via template")
        n4_bias(raw_anat, t1_n4)
        reg = brain_extract_via_template(str(t1_n4), str(t1_brain), str(t1_mask))
        inv_list = reg['invtransforms']
    else:
        # If T1 brain already exists, compute a quick registration to get transforms for MNI->T1 mask warps
        print(f"[{sub} {ses}] Computing T1->MNI transforms (for mask warps)")
        reg = t1_to_mni(str(t1_brain))
        inv_list = reg['invtransforms']

    # Warp MNI cerebellar GM mask to T1
    if not MNI_CEREB_MASK.exists():
        raise FileNotFoundError(f"Missing cerebellar mask {MNI_CEREB_MASK}")
    cereb_t1 = outdir / "cereb_gm_in_t1.nii.gz"
    if not cereb_t1.exists():
        print(f"[{sub} {ses}] Warping cerebellar GM mask MNI->T1")
        mni_cereb = ants.image_read(str(MNI_CEREB_MASK))
        t1_img = ants.image_read(str(t1_brain))
        warp_mni_mask_to_t1(mni_cereb, t1_img, inv_list, cereb_t1)

    # PET prep
    pet_avg = outdir / "pet_avg.nii.gz"
    if not pet_avg.exists():
        print(f"[{sub} {ses}] PET average (if 4D)")
        pet_avg_if4d(raw_pet, pet_avg)

    pet_in_t1 = outdir / "pet_in_t1.nii.gz"
    pet_tx = outdir / "pet_to_t1_0Rigid.mat"
    if not pet_in_t1.exists():
        print(f"[{sub} {ses}] PET->T1 rigid")
        pet_to_t1_rigid(pet_avg, t1_brain, pet_in_t1, pet_tx)

    pet_suvr = outdir / "pet_suvr_in_t1.nii.gz"
    if not pet_suvr.exists():
        print(f"[{sub} {ses}] SUVR")
        pet_img = nib.load(str(pet_in_t1)); pet = pet_img.get_fdata().astype(np.float32)
        ref = nib.load(str(cereb_t1)).get_fdata() > 0.5
        ref_mean = float(pet[ref].mean() + 1e-6)
        suvr = pet / ref_mean
        nib.Nifti1Image(suvr.astype(np.float32), pet_img.affine, pet_img.header).to_filename(str(pet_suvr))

def main():
    sel = pd.read_csv(LOG_DIR / "adni_selected_series.csv")
    # Find Subject/Session pairs that have both raw T1 and PET NIfTI already converted
    pairs = sel.pivot_table(index=["Subject","Session"], columns="which", values="series_dir", aggfunc="first").reset_index()
    pairs = pairs.dropna(subset=["T1","PET"])
    for _, row in pairs.iterrows():
        process_one(row["Subject"], str(row["Session"]))

if __name__ == "__main__":
    main()
