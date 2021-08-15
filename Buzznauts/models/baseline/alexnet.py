#!/usr/bin/env python
# coding=utf-8

import os.path as op
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import requests
import Buzznauts as buzz
import ast

# AlexNet Definition
__all__ = ['AlexNet', 'alexnet']

# Define AlexNet differently from torchvision code for better understanding


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.fc6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            )
        self.fc7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            )
        self.fc8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            )
        self.num_layers = 8

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out5_reshaped = out5.view(out5.size(0), 256 * 6 * 6)
        out6 = self.fc6(out5_reshaped)
        out7 = self.fc7(out6)
        out8 = self.fc8(out7)
        return out1, out2, out3, out4, out5, out6, out7, out8

    def get_key_list(self):
        key_list = ["conv1.0.weight", "conv1.0.bias",
                    "conv2.0.weight", "conv2.0.bias",
                    "conv3.0.weight", "conv3.0.bias",
                    "conv4.0.weight", "conv4.0.bias",
                    "conv5.0.weight", "conv5.0.bias",
                    "fc6.1.weight", "fc6.1.bias",
                    "fc7.1.weight", "fc7.1.bias",
                    "fc8.1.weight", "fc8.1.bias"]
        return key_list


def alexnet(pretrained=False, ckpth_urls=None, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    pretrained : bool
        if True, returns AlexNet pre-trained on ImageNet
    'ckpth_urls : dict[str] = str
        key -> model; value -> url to pre-trained weights.
        Default: ckpth_urls['alexnet'] =
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'

    kwargs : dict
        parameters to AlexNet model Class

    Returns
    -------
    model : AlexNet
        Pytorch instance of AlexNet model Class
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(ckpth_urls['alexnet']))

    return model


def load_alexnet(pretrained=False, custom_keys=False, **kwargs):
    """This function initializes an Alexnet and load its weights from a
    pretrained model. Since we redefined model in a different way we have to
    rename the weights that were in the pretrained checkpoint.

    Parameters
    ----------
    pretrained : bool
        if True, returns AlexNet pre-trained on ImageNet. Don't use if using
        custom checkpoint/state keys definition.

    custom_keys : bool
        if True, returns AlexNet pre-trained on ImageNet using custom
        checkpoint/state keys definition.

    kwargs : dict
        'ckpth_urls' : dict,
        'ckpth' : str
            filepath to pretrained AlexNet checkpoint
        Other entries are parameters to AlexNet

    Returns
    -------
    model : AlexNet
        Pytorch instance of the AlexNet model Class
    """
    if pretrained:
        ckpth_urls = kwargs.pop('ckpth_urls', None)
        ckpth_filepath = kwargs.pop('ckpth', './models/alexnet/alexnet.pth')
        if ckpth_urls is None:
            url ='https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
            ckpth_urls = {'alexnet': url}

        if custom_keys:
            # Don't use default state keys for pretrained weights
            model = alexnet(pretrained=False, **kwargs)

            # Download pretrained Alexnet from:
            # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
            # and save in the model directory
            chpth_dir = Path(ckpth_filepath).parent.absolute()
            if not op.exists(ckpth_dir):
                os.makedirs(ckpth_dir)
                r = requests.get(ckpth_urls['alexnet'])
                with open(ckpth_filepath, 'wb') as f:
                    f.write(r.content)

            ckpth = torch.load(ckpth_filepath,
                               map_location=lambda storage, loc: storage)

            # Remap checkpoint/state keys
            key_list = kwargs.pop('key_list', model.get_key_list())
            state_dict = {key_list[i]: v
                          for i, (k, v) in enumerate(ckpth.items())}

            # initialize model with pretrained weights
            model.load_state_dict(state_dict)

        else:
            model = alexnet(pretrained=True, ckpth_urls=ckpth_urls, **kwargs)
    else:
        model = alexnet(**kwargs)

    return model


def download_imagenet_labels(dst_file=None):
    if dst_file is None:
        buzz_root = Path(os.buzz.__path__[0]).parent.absolute()
        out_file = op.join(buzz_root, 'models/alexnet/imagenet_labels.txt')

    if not op.exists(dst_file):
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        r = requests.get(url)
        with open(dst_file, 'wb') as f:
            f.write(r.content)


def get_imagenet_labels(src_file):
    if not op.exists(src_file):
        download_imagenet_labels(src_file)
    with open(src_file, 'r') as f:
        contents = f.read()
        imagenet_dict = ast.literal_eval(contents)
    return imagenet_dict
