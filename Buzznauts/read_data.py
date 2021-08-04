#!/usr/bin/env python
# coding=utf-8

# Imports
import os
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from nilearn import datasets
from nilearn import plotting



# @title Utility functions for data loading
def save_dict(di_, filename_):
  with open(filename_, 'wb') as f:
      pickle.dump(di_, f)

def load_dict(filename_):
  with open(filename_, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    ret_di = u.load()
    # print(p)
    # ret_di = pickle.load(f)
  return ret_di


def visualize_activity(vid_id,sub, sub_fmri_dir, results_dir):

  # mapping the data to voxels and storing in a nifti file
  fmri_train_all,voxel_mask = get_fmri(sub_fmri_dir,"WB")
  visual_mask_3D = np.zeros((78,93,71))
  visual_mask_3D[voxel_mask==1]= fmri_train_all[vid_id,:]
  brain_mask = './example.nii'
  nii_save_path =  os.path.join(results_dir, 'vid_activity.nii')
  saveasnii(brain_mask,nii_save_path,visual_mask_3D)

  # visualizing saved nifti file
  plotting.plot_glass_brain(nii_save_path,
                          title='fMRI response',plot_abs=False,
                          display_mode='lyr',colorbar=True)


def get_fmri(fmri_dir, ROI):
  """This function loads fMRI data into a numpy array for to a given ROI.
  Parameters
  ----------
  fmri_dir : str
    path to fMRI data.
  ROI : str
    name of ROI.

  Returns
  -------
  np.array
    matrix of dimensions #train_vids x #repetitions x #voxels
    containing fMRI responses to train videos of a given ROI
  """

  # Loading ROI data
  ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
  ROI_data = load_dict(ROI_file)
  # averaging ROI data across repetitions
  ROI_data_train = np.mean(ROI_data["train"], axis=1)
  if ROI == "WB":
    voxel_mask = ROI_data['voxel_mask']

    return ROI_data_train, voxel_mask

  return ROI_data_train

def saveasnii(brain_mask,nii_save_path,nii_data):
  img = nib.load(brain_mask)
  nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
  nib.save(nii_img, nii_save_path)
