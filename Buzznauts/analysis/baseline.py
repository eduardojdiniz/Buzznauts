#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os
import os.path as op
import glob
from Buzznauts.models.baseline.alexnet import get_imagenet_labels
from Buzznauts.data.videodataframe import VideoFrameDataset, ImglistToTensor
from Buzznauts.utils import seed_worker, set_generator, set_seed
from Buzznauts.models.baseline.ols import OLS_pytorch
from Buzznauts.data.utils import visualize_activity_surf
from Buzznauts.utils import vectorized_correlation, get_fmri, saveasnii
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import torch
from torchvision import transforms as trn
import torch.nn.functional as F


def get_activations_and_save(model, frames_dir, imagenet_file,
                             activations_dir, device=None):
    """Generates Alexnet features and save them in a specified directory.

    Parameters
    ----------
    model : AlexNet
        AlexNet Pytorch model
    frames_dir : str
        path to video frames folder
    imagenet_file : str
        file with ImageNet class labels
    activations_dir : str
        save path for extracted features.
    device : str
        computational device (`cuda` if GPU is available, else `cpu`)
    """
    if device is None:
        device = set_device()

    # Path to file with the videos' metadata
    annotation_file = op.join(frames_dir, 'annotations.txt')

    # Resize and normalize transform for video frames
    preprocess = trn.Compose([
            ImglistToTensor(),
            trn.Resize(224),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = VideoFrameDataset(
        root_path=frames_dir,
        annotationfile_path=annotation_file,
        num_segments=16,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        random_shift=True,
        test_mode=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=set_generator())

    # Function to convert a tensor to numpy array
    to_numpy = lambda x: x.data.cpu().numpy().ravel()

    # Holds ImageNet top-5 class predictions for each video id
    #   key: vid id
    #   value: [(class, 1st probability), ..., (class, 5th probability)]
    predictions = {}

    # Each batch holds one video
    # loop through each video and save average activation for the frames
    for vid_id, (batch, filenames) in enumerate(tqdm(dataloader)):
        num_frames = batch.shape[1]
        batch = batch.to(device)
        x = model(batch.squeeze(0))
        # Save predictions to first frame by layer 7 of AlexNet
        predictions[vid_id] = get_alexnet_predictions(imagenet_file, x[7][0])
        activations = [to_numpy(torch.sum(feat, dim=0)) for feat in x]
        # Save the average activation for each layer
        for layer in range(len(activations)):
            filename = filenames[0] + "_layer_" + str(layer+1) + ".npy"
            save_path = op.join(activations_dir, filename)
            avg_layer_activation = activations[layer]/float(num_frames)
            np.save(save_path, avg_layer_activation)

    return predictions


def get_alexnet_predictions(src_file, output):
    imagenet_labels_dict = get_imagenet_labels(src_file)
    classes = [v.strip() for k, v in imagenet_labels_dict.items()]

    # sort the probability vector in descending order
    sorted, indices = torch.sort(output, descending=True)
    percentage = F.softmax(output, dim=0) * 100.0

    # Get the most probable first 5 classes the input belongs to
    predictions = [(classes[i], percentage[i].item()) for i in indices[:5]]
    return predictions


def do_PCA_and_save(activations_dir, save_dir, seed=None):
    """This function preprocesses Neural Network features using PCA and save
    the results in a specified directory.

    Parameters
    ----------
    activations_dir : str
        save path for extracted features.
    save_dir : str
        save path for extracted PCA features.
    seed : int
        Seed for the random generator
    """
    if seed is None:
        seed = set_seed()

    layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5', 'layer_6',
              'layer_7', 'layer_8']

    # Number of Principal Components
    n_components = 100

    if not op.exists(save_dir):
        os.makedirs(save_dir)

    for layer in tqdm(layers):
        regex = activations_dir + '/*' + layer + '.npy'
        activations_file_list = sorted(glob.glob(regex))
        feature_dim = np.load(activations_file_list[0])
        x = np.zeros((len(activations_file_list), feature_dim.shape[0]))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i, :] = temp
        x_train = x[:1000, :]
        x_test = x[1000:, :]

        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)
        ipca = PCA(n_components=n_components, random_state=seed)
        ipca.fit(x_train)

        x_train = ipca.transform(x_train)
        x_test = ipca.transform(x_test)
        train_save_path = op.join(save_dir, "train_" + layer)
        test_save_path = op.join(save_dir, "test_" + layer)
        np.save(train_save_path, x_train)
        np.save(test_save_path, x_test)


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
    features_train : np.array
        matrix of dimensions #train_vids x #pca_components containing
        activations of train videos

    features_test : np.array
        matrix of dimensions #test_vids x #pca_components containing
        activations of test videos

    """
    train_file = op.join(activations_dir, "train_" + layer_name + ".npy")
    test_file = op.join(activations_dir, "test_" + layer_name + ".npy")
    features_train = np.load(train_file)
    features_test = np.load(test_file)
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.fit_transform(features_test)

    return features_train, features_test


def predict_fmri_fast(features_train, features_test,
                      train_fmri, device=None):
    """This function fits a linear regressor using features_train and
    train_fmri, then returns the predicted fmri_pred_test using the fitted
    weights and features_test.

    Parameters
    ----------
    features_train : np.array
        matrix of dimensions #train_vids x #pca_components containing
        activations of train videos.
    features_test : np.array
        matrix of dimensions #test_vids x #pca_components containing
        activations of test videos
    train_fmri : np.array
        matrix of dimensions #train_vids x #voxels containing fMRI responses
        to train videos
    device : str
        computational device (cuda if GPU is available, else cpu)

    Returns
    -------
    fmri_pred_test: np.array
        matrix of dimensions #test_vids x #voxels containing predicted fMRI
        responses to test videos .

    """
    if device is None:
        device = set_device()

    reg = OLS_pytorch(device=device)
    reg.fit(features_train, train_fmri.T)
    fmri_pred_test = reg.predict(features_test)

    return fmri_pred_test


def perform_encoding(activations_dir, results_dir, fmri_dir, sub, layer,
                     ROI="WB", mode="val", visualize_results=True,
                     batch_size=1000, device=None, viz_dir=None):

    if device is None:
        device = set_device()

    track = "full_track" if ROI == "WB" else "mini_track"
    sub_fmri_dir = op.join(fmri_dir, track, sub)
    results_dir = op.join(results_dir, layer, track, sub)
    if not op.exists(results_dir):
        os.makedirs(results_dir)

    features_train, features_test = get_activations(activations_dir, layer)
    if track == "full_track":
        fmri_data, voxel_mask = get_fmri(sub_fmri_dir, ROI)
    else:
        fmri_data = get_fmri(sub_fmri_dir, ROI)
    num_voxels = fmri_data.shape[1]

    if mode == 'val':
        # We use first 900 videos as training and rest as validation
        features_test = features_train[900:, :]
        features_train = features_train[:900, :]
        fmri_train = fmri_data[:900, :]
        fmri_test = fmri_data[900:, :]
        pred_fmri = np.zeros_like(fmri_test)
        pred_fmri_save_path = op.join(results_dir, ROI + '_val.npy')
    else:
        fmri_train = fmri_data
        num_test_videos = 102
        pred_fmri = np.zeros((num_test_videos, num_voxels))
        pred_fmri_save_path = op.join(results_dir, ROI + '_test.npy')

    i = 0
    while i < num_voxels - batch_size:
        j = i + batch_size
        pred_fmri[:, i:j] = predict_fmri_fast(features_train,
                                                features_test,
                                                fmri_train[:, i:j],
                                                device=device)
        i = j
    pred_fmri[:, i:] = predict_fmri_fast(features_train,
                                            features_test,
                                            fmri_train[:, i:i + batch_size],
                                            device=device)

    np.save(pred_fmri_save_path, pred_fmri)

    if mode == 'val':
        score = vectorized_correlation(fmri_test, pred_fmri)
        mean_score = score.mean()

        # result visualization for whole brain (full_track)
        if track == "full_track" and visualize_results:
            viz_dir = op.join(viz_dir, layer, track, sub)
            if not op.exists(viz_dir):
                os.makedirs(viz_dir)
            brain_mask = op.join(fmri_dir, 'example.nii')
            nii_save_path = op.join(viz_dir, ROI + '_val.nii')
            html_save_path = op.join(viz_dir, ROI + '_val.html')

            title = f'Predicted fMRI response for {sub} on validation videos'
            view_args = {'brain_mask': brain_mask,
                         'nii_save_path': nii_save_path,
                         'html_save_path': html_save_path,
                         'score': score,
                         'voxel_mask': voxel_mask,
                         'title': title}

            view_html = visualize_activity_surf(sub, **view_args)
            #view.open_in_browser()

            return mean_score, view_html
        else:
            return mean_score
