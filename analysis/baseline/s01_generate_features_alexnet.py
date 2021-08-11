#!/usr/bin/env python
# coding=utf-8

import torch
import numpy as np
import random
import argparse
import os
import os.path as op
import glob
from Buzznauts.models.baseline.alexnet import load_alexnet
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
from decord import VideoReader
from decord import cpu
from torchvision import transforms as trn

seed = 42
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)


def sample_video_from_mp4(file, num_frames=16):
    """This function takes a mp4 video file as input and returns a list of
    uniformly sampled frames (PIL Image).

    Parameters
    ----------
    file : str
        path to mp4 video file
    num_frames : int
        how many frames to select using uniform frame sampling.

    Returns
    -------
    images : list[PIL.Image]
        list of PIL Images
    num_frames: int
        number of frames extracted

    """
    images = list()
    vr = VideoReader(file, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames-1, num_frames, dtype=np.int32)
    for seg_ind in indices:
        images.append(Image.fromarray(vr[seg_ind].asnumpy()))

    return images, num_frames


def get_activations_and_save(model, video_list, activations_dir):
    """Generates Alexnet features and save them in a specified directory.

    Parameters
    ----------
    model :
        model : AlexNet
            AlexNet Pytorch model
    video_list : list
        the list contains path to all videos.
    activations_dir : str
        save path for extracted features.

    """
    # Resize and normalize transform for images
    resize_normalize = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # for all videos in the list
    for video_file in tqdm(video_list):
        vid, num_frames = sample_video_from_mp4(video_file)
        video_filename = os.path.split(video_file)[-1].split(".")[0]
        activations = []
        # for all frames in video generate and save activations
        for frame, img in enumerate(vid):
            input_img = torch.Tensor(resize_normalize(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img = input_img.cuda()
            x = model.forward(input_img)
            for i, feat in enumerate(x):
                if frame == 0:
                    activations.append(feat.data.cpu().numpy().ravel())
                else:
                    activations[i] += feat.data.cpu().numpy().ravel()
        for layer in range(len(activations)):
            filename = video_filename + "_layer_" + str(layer+1) + ".npy"
            save_path = os.path.join(activations_dir, filename)
            avg_layer_activation = activations[layer]/float(num_frames)
            np.save(save_path, avg_layer_activation)


def do_PCA_and_save(activations_dir, save_dir):
    """This function preprocesses Neural Network features using PCA and save
    the results in a specified directory.

    Parameters
    ----------
    activations_dir : str
        save path for extracted features.
    save_dir : str
        save path for extracted PCA features.

    """
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


def main():
    buzz_root = '/home/dinize@acct.upmchs.net/proj/Buzznauts'
    description = 'Feature Extraction from Alexnet and preprocessing using PCA'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-vdir', '--video_data_dir',
                        help='video data directory',
                        default=op.join(buzz_root, 'data/stimuli/videos'),
                        type=str)
    parser.add_argument('-sdir', '--save_dir',
                        help='saves processed features',
                        default=op.join(buzz_root, 'models/baseline'),
                        type=str)
    args = vars(parser.parse_args())

    save_dir = args['save_dir']
    if not op.exists(save_dir):
        os.makedirs(save_dir)

    video_dir = args['video_data_dir']
    video_list = sorted(glob.glob(video_dir + '/*.mp4'))
    print('Total Number of Videos: ', len(video_list))

    # Petrained Alexnet from:
    # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
    checkpoint_path = op.join(save_dir, "alexnet.pth")
    url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
    kwargs = {'ckpth_urls': {'alexnet': url}, 'ckpth': checkpoint_path}

    # download pretrained model and save in the current directory
    model = load_alexnet(pretrained=True, custom_keys=True, **kwargs)

    # get and save activations
    activations_dir = op.join(save_dir, 'activations')
    if not op.exists(activations_dir):
        os.makedirs(activations_dir)
    print("-------------------Saving activations ----------------------------")
    get_activations_and_save(model, video_list, activations_dir)

    # preprocessing using PCA and save
    pca_dir = op.join(activations_dir, 'pca_100')
    print("-----------------------Performing  PCA----------------------------")
    do_PCA_and_save(activations_dir, pca_dir)


if __name__ == "__main__":
    main()
