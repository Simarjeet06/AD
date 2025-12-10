# E:\adni_python\code\convert_pet_dynamic_safe.py
from pathlib import Path
import os
import pydicom
import nibabel as nib
import numpy as np
from collections import defaultdict

def load_headers(dcm_dir):
    files = [str(Path(dcm_dir)/f) for f in os.listdir(dcm_dir) if f.lower().endswith(".dcm")]
    hdrs = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            hdrs.append((f, ds))
        except Exception:
            pass
    return hdrs

def group_by_timepoint(hdrs):
    groups = defaultdict(list)
    for f, ds in hdrs:
        tpi = getattr(ds, "TemporalPositionIdentifier", None)
        if tpi is None:
            tkey = getattr(ds, "ContentTime", None) or getattr(ds, "AcquisitionTime", None)
        else:
            tkey = int(tpi)
        z = getattr(ds, "InstanceNumber", None)
        groups[tkey].append((z, f))
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: (x[0] if x is not None else 0))
    return groups

def stacks_to_nifti(groups, out_avg_nii):
    volumes = []
    aff_ref = None
    shape_ref = None
    for k in sorted(groups.keys()):
        files = [f for _, f in groups[k]]
        slicestack = []
        ds_last = None
        for f in files:
            ds = pydicom.dcmread(f, force=True)
            ds_last = ds
            arr = ds.pixel_array.astype(np.float32)
            slicestack.append(arr)
        vol = np.stack(slicestack, axis=-1)
        # Build a simple affine from last slice header
        try:
            ipp = np.array(ds_last.ImagePositionPatient, dtype=float)
            iop = np.array(ds_last.ImageOrientationPatient, dtype=float).reshape(2,3)
            px, py = map(float, ds_last.PixelSpacing)
            dz = float(getattr(ds_last, "SpacingBetweenSlices", getattr(ds_last, "SliceThickness", 1.0)))
            xdir, ydir = iop
            zdir = np.cross(xdir, ydir)
            aff = np.eye(4, dtype=float)
            aff[:3,0] = xdir * px
            aff[:3,1] = ydir * py
            aff[:3,2] = zdir * dz
            aff[:3,3] = ipp
        except Exception:
            aff = np.eye(4, dtype=float)

        if shape_ref is None:
            shape_ref = vol.shape
            aff_ref = aff
        if vol.shape == shape_ref:
            volumes.append(vol)

    if len(volumes) == 0:
        raise RuntimeError("No consistent timepoints to average")

    avg = np.mean(np.stack(volumes, axis=0), axis=0)
    nib.Nifti1Image(avg.astype(np.float32), aff_ref).to_filename(str(out_avg_nii))

def convert_pet_series_safe(series_dir, out_nii):
    hdrs = load_headers(series_dir)
    if not hdrs:
        raise RuntimeError("No DICOMs")
    groups = group_by_timepoint(hdrs)
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    stacks_to_nifti(groups, out_nii)