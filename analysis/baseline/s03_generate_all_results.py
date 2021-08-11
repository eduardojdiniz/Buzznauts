#!/usr/bin/env python
# coding=utf-8

import os
import os.path as op
import argparse


def main():
    description = 'Generates predictions for all subs and all ROIs for a track'
    parser = argparse.ArgumentParser(description=description)

    buzz_root = '/home/dinize@acct.upmchs.net/proj/Buzznauts'
    _help = 'mini_track for all ROIs, full_track for whole brain (WB)'
    parser.add_argument('-t', '--track',
                        help=_help,
                        default='mini_track',
                        type=str)
    parser.add_argument('-fd', '--fmri_dir',
                        help='directory containing fMRI activity',
                        default=os.path.join(buzz_root, 'data/fmri'),
                        type=str)
    _help = 'layer from which activations will be used to train & predict fMRI'
    parser.add_argument('-l', '--layer',
                        help=_help,
                        default='layer_5',
                        type=str)

    args = vars(parser.parse_args())
    track = args['track']
    fmri_dir = args['fmri_dir']
    layer = args['layer']

    model_scripts = op.join(buzz_root, 'analysis/baseline')
    encoding_script = op.join(model_scripts, 's02_perform_encoding.py')

    if track == 'full_track':
        ROIs = ['WB']
    else:
        ROIs = ['LOC', 'FFA', 'STS', 'EBA', 'PPA', 'V1', 'V2', 'V3', 'V4']

    num_subs = 10
    subs = []
    for s in range(num_subs):
        subs.append('sub' + str(s+1).zfill(2))

    for roi in ROIs:
        for sub in subs:
            cmd_string = 'python ' + encoding_script + ' --roi ' + roi +  \
                ' --sub ' + sub + ' -fd ' + fmri_dir + ' --layer ' + layer + \
                ' --mode  test'
            print("Starting ROI: ", roi, "sub: ", sub)
            os.system(cmd_string)
            print("Completed ROI: ", roi, "sub: ", sub)
            print("----------------------------------------------------------")


if __name__ == "__main__":
    main()
