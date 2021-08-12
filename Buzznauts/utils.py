#!/usr/bin/env python
# coding=utf-8

import nibabel as nib
import pickle
import os.path as op
import numpy as np
import warnings

# Ignore FutureWarnings from nilearn
with warnings.catch_warnings():
    # ignore all caught warnings
    warnings.filterwarnings("ignore")
    from nilearn import plotting


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
    return ret_di


def saveasnii(brain_mask, nii_save_path, nii_data):
    img = nib.load(brain_mask)
    print(img.shape)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)


def vectorized_correlation(x, y):
    dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True)+1e-8
    y_std = y.std(axis=dim, keepdims=True)+1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.ravel()


def get_fmri(fmri_dir, ROI):
    """This function loads fMRI data into a numpy array for a given ROI.

    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    ROI : str
        name of ROI.

    Returns
    -------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels containing
        fMRI responses to train videos of a given ROI

    """
    # Loading ROI data
    ROI_file = op.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions
    ROI_data_train = np.mean(ROI_data["train"], axis=1)
    if ROI == "WB":
        voxel_mask = ROI_data['voxel_mask']
        return ROI_data_train, voxel_mask

    return ROI_data_train


def visualize_activity(sub, **kwargs):
    vid_id = kwargs.pop('vid_id', 0)

    buzz_root = '/home/dinize@acct.upmchs.net/proj/Buzznauts'
    fmri_dir = op.join(buzz_root, 'data/fmri')
    track_dir = kwargs.pop('track_dir', op.join(fmri_dir, 'full_track'))

    brain_mask = kwargs.pop('brain_mask', op.join(fmri_dir, 'example.nii'))

    results_dir = op.join(buzz_root, 'models/baseline/results')
    nii_save_path =  op.join(results_dir, 'vid_activity.nii')
    nii_save_path = kwargs.pop('nii_save_path', nii_save_path)

    # Plotting parameters
    threshold = kwargs.pop('threshold', None)
    surf_mesh = kwargs.pop('surf_mesh', 'fsaverage')
    title = kwargs.pop('title', 'fMRI response for sub' + sub)
    colobar = kwargs.pop('colobar', True)

    # Subject parameters
    score = kwargs.pop('score', None)
    voxel_mask = kwargs.pop('voxel_mask', None)

    visual_mask_3D = np.zeros((78,93,71))
    if score is None:
        sub_fmri_dir = op.join(track_dir, sub)
        fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir, "WB")
        visual_mask_3D[voxel_mask==1] = fmri_train_all[vid_id, :]
    else:
        visual_mask_3D[voxel_mask==1] = score

    saveasnii(brain_mask, nii_save_path, visual_mask_3D)

    view = plotting.view_img_on_surf(nii_save_path,
                                    threshold = threshold,
                                    surf_mesh = surf_mesh,
                                    title = title,
                                    colorbar = colobar)
    return view
