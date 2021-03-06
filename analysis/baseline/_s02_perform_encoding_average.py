#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os
import os.path as op
import argparse
from sklearn.preprocessing import StandardScaler
import torch
from Buzznauts.models.baseline.ols import OLS_pytorch
from Buzznauts.utils import vectorized_correlation, visualize_activity
from Buzznauts.utils import load_dict, saveasnii, get_fmri
from tqdm import tqdm


def get_activations(activations_dir, layer_name):
    """This function loads neural network features/activations (preprocessed
    using PCA) into a numpy array according to a given layer.

    Parameters
    ----------
    activations_dir : str
        Path to PCA processed Neural Network features
    layer_name : str
        which layer of the neural network to load,

    Returns
    -------
    train_activations : np.array
        matrix of dimensions #train_vids x #pca_components containing
        activations of train videos

    test_activations : np.array
        matrix of dimensions #test_vids x #pca_components containing
        activations of test videos

    """
    train_file = op.join(activations_dir, "train_" + layer_name + ".npy")
    test_file = op.join(activations_dir, "test_" + layer_name + ".npy")
    train_activations = np.load(train_file)
    test_activations = np.load(test_file)
    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)

    return train_activations, test_activations


def predict_fmri_fast(train_activations, test_activations,
                      train_fmri, use_gpu=False):
    """This function fits a linear regressor using train_activations and
    train_fmri, then returns the predicted fmri_pred_test using the fitted
    weights and test_activations.

    Parameters
    ----------
    train_activations : np.array
        matrix of dimensions #train_vids x #pca_components containing
        activations of train videos.
    test_activations : np.array
        matrix of dimensions #test_vids x #pca_components containing
        activations of test videos
    train_fmri : np.array
        matrix of dimensions #train_vids x #voxels containing fMRI responses
        to train videos
    use_gpu : bool
        Description of parameter `use_gpu`.

    Returns
    -------
    fmri_pred_test: np.array
        matrix of dimensions #test_vids x #voxels containing predicted fMRI
        responses to test videos .

    """
    reg = OLS_pytorch(use_gpu)
    reg.fit(train_activations, train_fmri.T)
    fmri_pred_test = reg.predict(test_activations)
    return fmri_pred_test

def get_score(fmri_test, pred_fmri)
    score = vectorized_correlation(fmri_test, pred_fmri)
    return round(score.mean(), 3)

def main():
    description = 'Encoding model analysis for Algonauts 2021'
    parser = argparse.ArgumentParser(description=description)

    buzz_root = '/home/dinize@acct.upmchs.net/proj/Buzznauts'
    baseline = op.join(buzz_root, 'models/baseline')
    parser.add_argument('-rd', '--result_dir',
                        help='saves predicted fMRI activity',
                        default=op.join(baseline, 'results'),
                        type=str)
    parser.add_argument('-ad', '--activation_dir',
                        help='directory containing DNN activations',
                        default=op.join(baseline, 'activations'),
                        type=str)
    parser.add_argument('-model', '--model',
                        help='model under which predicted fMRI will be saved',
                        default='alexnet',
                        type=str)
    _help = 'layer from which activations will be used to train & predict fMRI'
    parser.add_argument('-l', '--layer',
                        help=_help,
                        default='layer_5',
                        type=str)
    parser.add_argument('-r', '--roi',
                        help='brain region from which fMRI data will be used',
                        default='EBA',
                        type=str)
    _help = 'test or val, val returns mean correlation ' + \
        'by using 10% of training data for validation'
    parser.add_argument('-m', '--mode',
                        help=_help,
                        default='val',
                        type=str)
    parser.add_argument('-fd', '--fmri_dir',
                        help='directory containing fMRI activity',
                        default=op.join(buzz_root, 'data/fmri'),
                        type=str)
    parser.add_argument('-v', '--visualize',
                        help='visualize whole brain in MNI space or not',
                        default=True,
                        type=bool)
    _help = 'number of voxel to fit at one time in case of memory constraints'
    parser.add_argument('-b', '--batch_size',
                        help=_help,
                        default=1000,
                        type=int)
    args = vars(parser.parse_args())

    mode = args['mode']
    ROI = args['roi']
    model = args['model']
    layer = args['layer']
    visualize_results = args['visualize']
    batch_size = args['batch_size']

    # List of subjects
    subs = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05',
            'sub06', 'sub07', 'sub08', 'sub09', 'sub10']
    # Subjects' scores
    sub_scores = {sub: 0.0 for sub in subs}

    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    if ROI == "WB":
        track = "full_track"
    else:
        track = "mini_track"

    activation_dir = op.join(args['activation_dir'], 'pca_100')
    fmri_dir = op.join(args['fmri_dir'], track)

    sub_fmri_dir = op.join(fmri_dir, sub)
    results_dir = op.join(args['result_dir'], model, layer, track, sub)
    if not op.exists(results_dir):
        os.makedirs(results_dir)

    print("ROi is : ", ROI)

    train_activations, test_activations = get_activations(activation_dir,
                                                          layer)
    if track == "full_track":
        fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir, ROI)
    else:
        fmri_train_all = get_fmri(sub_fmri_dir, ROI)
    num_voxels = fmri_train_all.shape[1]

    if mode == 'val':
        # Here as an example we use first 900 videos as training and rest of
        # the videos as validation
        test_activations = train_activations[900:, :]
        train_activations = train_activations[:900, :]
        fmri_train = fmri_train_all[:900, :]
        fmri_test = fmri_train_all[900:, :]
        pred_fmri = np.zeros_like(fmri_test)
        pred_fmri_save_path = op.join(results_dir, ROI + '_val.npy')
    else:
        fmri_train = fmri_train_all
        num_test_videos = 102
        pred_fmri = np.zeros((num_test_videos, num_voxels))
        pred_fmri_save_path = op.join(results_dir, ROI + '_test.npy')

    print("number of voxels is ", num_voxels)
    i = 0
    with tqdm(total=100) as pbar:
        while i < num_voxels - batch_size:
            j = i + batch_size
            pred_fmri[:, i:j] = predict_fmri_fast(train_activations,
                                                test_activations,
                                                fmri_train[:, i:j],
                                                use_gpu=use_gpu)
            i = j
            pbar.update((100*i) // num_voxels)
    pred_fmri[:, i:] = predict_fmri_fast(train_activations,
                                         test_activations,
                                         fmri_train[:, i:i + batch_size],
                                         use_gpu=use_gpu)

    if mode == 'val':
        for sub in subs:
            sub_scores[sub] = get_mean_score(fmri_test, pred_fmri)

        print("Mean correlation for ROI : ", ROI, "in ", sub, " is :", score)

        # result visualization for whole brain (full_track)
        if track == "full_track" and visualize_results:
            brain_mask = op.join(buzz_root, 'data/fmri/example.nii')
            nii_save_path = op.join(results_dir, ROI + '_val.nii')

            view_args = {'brain_mask': brain_mask,
                         'nii_save_path': nii_save_path,
                         'score': score,
                         'voxel_mask': voxel_mask}

            view = visualize_activity(sub, **view_args)
            view_save_path = op.join(results_dir, ROI + '_val.html')
            view.save_as_html(view_save_path)
            print("Results saved in this directory: ", results_dir)
            view.open_in_browser()

    np.save(pred_fmri_save_path, pred_fmri)

    print("ROI done : ", ROI)


if __name__ == "__main__":
    main()
