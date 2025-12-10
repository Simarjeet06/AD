# # E:\adni_python\code\make_cerebellum_mask_aal.py
# from pathlib import Path
# import numpy as np
# import nibabel as nib

# # Paths in your project
# PROJECT = Path(r"E:\adni_python")
# AAL_PATH = PROJECT / "atlas" / "atlases" / "atlas_aal.nii.gz"  # adjust if your filename differs
# OUT_MASK = PROJECT / "atlas" / "cereb_gm_mask_mni.nii.gz"

# # Common AAL cerebellar label IDs (AAL2 range 91..116).
# # If this yields 0 voxels, we’ll print unique labels to discover your IDs.
# AAL_CEREB_IDS = list(range(91, 117))

# def main():
#     img = nib.load(str(AAL_PATH))
#     data = img.get_fdata()

#     mask = np.isin(data, AAL_CEREB_IDS).astype(np.uint8)
#     if mask.sum() == 0:
#         # Fallback: print unique labels so we can map the right IDs for your file
#         uniq = np.unique(data.astype(np.int32))
#         print("Mask is empty. Unique labels found in your AAL file (first 120 shown):")
#         print(uniq[:120])
#         print("Share these values and I’ll give you the exact cerebellar IDs for your file.")
#         return

#     nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(OUT_MASK))
#     print(f"Saved cerebellar mask: {OUT_MASK} | voxels: {int(mask.sum())}")

# if __name__ == "__main__":
#     main()

# E:\adni_python\code\make_cerebellum_mask_aal.py
from pathlib import Path
import numpy as np
import nibabel as nib

PROJECT = Path(r"E:\adni_python")
AAL_PATH = PROJECT / "atlas" / "atlases" / "atlas_aal.nii.gz"  # adjust if filename differs
OUT_MASK = PROJECT / "atlas" / "cereb_gm_mask_mni.nii.gz"

# Cerebellar gray-matter labels for your AAL build (four-digit codes)
AAL_CEREB_GM_IDS = [
    9001, 9011, 9021, 9031, 9041, 9051, 9061, 9071, 9081,  # left
    9002, 9012, 9022, 9032, 9042, 9052, 9062, 9072, 9082   # right
]

def main():
    img = nib.load(str(AAL_PATH))
    data = img.get_fdata()
    mask = np.isin(data, AAL_CEREB_GM_IDS).astype(np.uint8)

    if mask.sum() == 0:
        uniq = np.unique(data.astype(np.int32))
        print("Mask is empty. Unique labels (first 200):")
        print(uniq[:200])
        return

    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(OUT_MASK))
    print(f"Saved cerebellar GM mask: {OUT_MASK} | voxels: {int(mask.sum())}")

if __name__ == "__main__":
    main()
