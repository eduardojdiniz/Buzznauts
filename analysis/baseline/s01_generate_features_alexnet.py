#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import os.path as op
from Buzznauts.models.baseline.alexnet import load_alexnet
from Buzznauts.utils import set_seed, set_device
from Buzznauts.analysis.baseline import get_activations_and_save
from Buzznauts.analysis.baseline import do_PCA_and_save


def main():
    buzz_root = '/home/dinize@acct.upmchs.net/proj/Buzznauts'
    description = 'Feature Extraction from Alexnet and preprocessing using PCA'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-vdir', '--video_frames_dir',
                        help='video frames data directory',
                        default=op.join(buzz_root, 'data/stimuli/frames'),
                        type=str)
    parser.add_argument('-sdir', '--save_dir',
                        help='saves processed features',
                        default=op.join(buzz_root, 'models/baseline'),
                        type=str)
    args = vars(parser.parse_args())

    save_dir = args['save_dir']
    if not op.exists(save_dir):
        os.makedirs(save_dir)

    frames_dir = args['video_frames_dir']

    # Call set_seed to ensure reproducibility
    seed = set_seed()

    # Set computational device (cuda is GPU is available, else cpu)
    device = set_device()

    # Petrained Alexnet from:
    # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
    checkpoint_path = op.join(save_dir, "alexnet.pth")
    url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
    kwargs = {'ckpth_urls': {'alexnet': url}, 'ckpth': checkpoint_path}

    # download pretrained model and save in the current directory
    model = load_alexnet(pretrained=True, custom_keys=True, **kwargs)
    model.to(device)
    model.eval()

    # get and save activations
    activations_dir = op.join(save_dir, 'activations')
    if not op.exists(activations_dir):
        os.makedirs(activations_dir)

    print("-------------------Saving activations ----------------------------")
    imagenet_file = op.join(save_dir, 'imagenet_labels.txt')
    _ = get_activations_and_save(model, frames_dir, activations_dir
                                 imagenet_file, device=device)

    # preprocessing using PCA and save
    pca_dir = op.join(activations_dir, 'pca_100')
    print("-----------------------Performing  PCA----------------------------")
    do_PCA_and_save(activations_dir, pca_dir, seed=seed)


if __name__ == "__main__":
    main()
