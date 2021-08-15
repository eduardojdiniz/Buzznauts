#!/usr/bin/env python
# coding=utf-8

import nibabel as nib
import pickle
import os.path as op
import numpy as np
import random
import torch

# Ignore FutureWarnings from nilearn
import warnings
with warnings.catch_warnings():
    # ignore all caught warnings
    warnings.filterwarnings("ignore")
    from nilearn import plotting


def set_seed(seed=None, seed_torch=True):
    """Set seed of random number generators to limit the number of sources of
    nondeterministic behavior for a specific platform, device, and PyTorch
    release. For more information, see
    https://pytorch.org/docs/stable/notes/randomness.html.

    Parameters
    ----------
    seed: int
        Seed for random number generators.
        Default: 2**32
    seed_torch: bool
        If we will set the seed for torch random number generators.
        Default: True

    Returns
    -------
    seed : int
        Seed used for random number generators
    """
    if seed is None:
        seed = np.random.choice(2 ** 32)

    # Set python seed for custom operators
    random.seed(seed)

    # Set seed for the global NumPy RNG if any of the libraries rely on NumPy
    np.random.seed(seed)

    if seed_torch:
        # Seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        # Set cuDNN to deterministically select a convolution algorithm
        torch.backends.cudnn.benchmark = False
        # Ensure that the cuDNN convolution algorithm is deterministic
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')

    return seed


def seed_worker(worker_id):
    """Set seed for Dataloader. DataLoader will reseed workers the "Randomness
    in multi-process data loading" algorithm. Use `worker_init_fn()` to
    preserve reproducibility. For more information, see
    https://pytorch.org/docs/stable/notes/randomness.html.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_generator(seed=0):
    """Set seed for Dataloader generators. DataLoader will reseed workers the
    "Randomness in multi-process data loading" algorithm. Use `generator` to
    preserve reproducibility. For more information, see
    https://pytorch.org/docs/stable/notes/randomness.html.

    Parameters
    ----------
    seed: int
        Seed for random number generators.
        Default: 0

    Returns
    -------
    generator: torch.Generator
        Seeded torch generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def set_device():
    """
    Set device (GPU or CPU).

    Returns
    -------
    device: str
        The device to be used, either `cpu` or `gpu`

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Inform the user if torch  uses GPU or CPU.
    if device != "cuda":
        print("GPU is not enabled.")
    else:
        print("GPU is enabled.")

    return device


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
