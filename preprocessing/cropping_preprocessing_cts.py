import os, gc, json, shutil, subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import SimpleITK as sitk
from scipy import ndimage
import nibabel as nib
import torch.nn.functional as F
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator

BASE_TMP = Path("Temperory processing file storage directory")
BASE_TMP.mkdir(parents = True, exist_ok = True)

class Segment_cropped_lung():
    def __init__(self, volume_path, margin):
        self.ct_path        = volume_path
        self.margin_mm      = margin
    def segment(self):
        LUNG_CLASSES = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
        seg_img      = totalsegmentator(input=str(self.ct_path), fast=True, roi_subset=LUNG_CLASSES, device="gpu:0", nr_thr_saving=1)
        return seg_img
    def cropping(self):
        mask_img  = self.segment()
        ct_img    = nib.load(self.ct_path)
        ct        = np.asanyarray(ct_img.dataobj)
        if ct.ndim == 4:
            ct = ct[..., 0]
        mask   = mask_img.get_fdata() > 0
        coords = np.argwhere(mask)
        axcodes              = nib.aff2axcodes(ct_img.affine)
        si_axes              = [i for i, code in enumerate(axcodes) if code in ("S", "I")]
        si_axis              = si_axes[0]
        voxel_sizes          = np.array(ct_img.header.get_zooms()[:3])
        margin_vox_si        = int(np.ceil(self.margin_mm / voxel_sizes[si_axis]))
        si_min               = coords[:, si_axis].min()
        si_max               = coords[:, si_axis].max() + 1
        si_min               = max(si_min - margin_vox_si, 0)
        si_max               = min(si_max + margin_vox_si, ct.shape[si_axis])
        slices               = [slice(None), slice(None), slice(None)]
        slices[si_axis]      = slice(si_min, si_max)
        slices               = tuple(slices)
        ct_crop              = ct[slices]
        mask_crop            = mask[slices]
        start_voxel          = np.array([0, 0, 0])
        start_voxel[si_axis] = si_min
        new_affine           = ct_img.affine.copy()
        new_affine[:3, 3]    = nib.affines.apply_affine(ct_img.affine, start_voxel)
        new_header           = ct_img.header.copy()

        new_header.set_data_shape(ct_crop.shape)
        new_header.set_zooms(ct_img.header.get_zooms()[:3])
        new_header.set_data_dtype(ct_img.get_data_dtype())
        cropped_img = nib.Nifti1Image(ct_crop, new_affine, new_header)
        cropped_img.set_qform(new_affine, code=1)
        cropped_img.set_sform(new_affine, code=1)
        return cropped_img

class CTImageProcessing:
    def __init__(self, split, target_spacing, WL, WW, final_shape_dhw, croping_margin):
        self.target_spacing_zyx            = target_spacing
        self.WL                            = WL
        self.WW                            = WW
        self.final_shape_dhw               = final_shape_dhw
        self.croping_margin                = croping_margin
        self.split                         = split
        with open("Path to initial 3D CT scan volumes", "r") as js:
            self.nii_paths = json.load(js)
    def resample_to_spacing(self, image):
        original_spacing_xyz = image.GetSpacing()
        original_size_xyz    = image.GetSize()
        target_spacing_xyz   = (self.target_spacing_zyx[2], self.target_spacing_zyx[1], self.target_spacing_zyx[0])
        new_size_xyz         = [int(round(original_size_xyz[i] * (original_spacing_xyz[i] / target_spacing_xyz[i]))) for i in range(3)]
        resampler            = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing_xyz)
        resampler.SetSize(new_size_xyz)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1024.0)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampled = resampler.Execute(image)
        return resampled
    def window_and_normalize(self, volume):
        lower  = self.WL - self.WW / 2.0
        upper  = self.WL + self.WW / 2.0
        volume = np.clip(volume, lower, upper)
        volume = (volume - lower) / self.WW
        volume = volume * 2.0 - 1.0
        volume = np.clip(volume, -1.0, 1.0).astype(np.float32)
        return volume
    def resize_to_fixed_shape(self, volume):
        target_D, target_H, target_W = self.final_shape_dhw
        x                            = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]             
        x                            = torch.nn.functional.interpolate(x, size=(target_D, target_H, target_W), mode="trilinear", align_corners=False)
        x                            = x.squeeze(0).squeeze(0)                   # [target_D, target_H, target_W]
        return x.cpu().numpy().astype(np.float32)
    def preprocess_cropped_resized_volume(self):
        for case in tqdm(self.nii_paths):
            nii_path     = self.nii_paths[case]
            case_tmp_dir = BASE_TMP / f"{self.split}_{case}"
            case_tmp_dir.mkdir(parents = True, exist_ok = True)
            tmp_path     = case_tmp_dir / "tmp.nii.gz"
            try:
                # cropping original volume around lung and saving it temp
                cropping_class = Segment_cropped_lung(nii_path, self.croping_margin)
                cropped_img    = cropping_class.cropping()
                nib.save(cropped_img, tmp_path)
                # resample and convert to numpy
                image = sitk.ReadImage(str(tmp_path))
                image = sitk.DICOMOrient(image, "LPS")
                tmp_path.unlink(missing_ok = True)
                image     = self.resample_to_spacing(image)
                volume_hu = sitk.GetArrayFromImage(image).astype(np.float32)
                # Windowing and normalize to [-1, 1]
                volume_norm  = self.window_and_normalize(volume_hu)
                # crop or pad to fix shape for batching
                volume_fixed = self.resize_to_fixed_shape(volume_norm)
                torch.save(volume_fixed, f"Path to save processed volume/{self.split}/{case}.pt")
                del image, volume_hu, volume_norm, volume_fixed, cropped_img
                gc.collect()
            finally:
                shutil.rmtree(case_tmp_dir, ignore_errors = True)

if __name__ == "__main__":
    target_shape = (256, 256, 256) # (D, H, W)
    for split in ['train', 'val', 'test']:
        processor = CTImageProcessing(split, (1.5, 0.75, 0.75), -600.0, 1500.0, target_shape, 12)
        processor.preprocess_cropped_resized_volume()