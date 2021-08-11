#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os.path as op
import argparse
import zipfile
from Buzznauts.utils import save_dict

def main():
    buzz_root = '/home/dinize@acct.upmchs.net/proj/Buzznauts'
    description = 'Prepares submission for Algonauts 2021'
    parser = argparse.ArgumentParser(description=description)

    model = op.join(buzz_root, 'models/baseline')
    parser.add_argument('-rd', '--result_dir',
                        help='contains predicted fMRI activity',
                        default = op.join(model, 'results/alexnet/layer_5'),
                        type=str)
    _help= 'mini_track for all ROIs, full_track for whole brain (WB)'
    parser.add_argument('-t', '--track',
                        help=_help,
                        default='mini_track',
                        type=str)
    args = vars(parser.parse_args())

    track = args['track']
    result_dir = args['result_dir']

    if track == 'full_track':
        ROIs = ['WB']
    else:
        ROIs = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']

    num_subs = 10
    subs=[]
    for s in range(num_subs):
        subs.append('sub' + str(s+1).zfill(2))

    results = {}
    for ROI in ROIs:
        ROI_results = {}
        for sub in subs:
            ROI_result_file = op.join(result_dir, track, sub, ROI+"_test.npy")
            print("Result file path: ", ROI_result_file)
            if not op.exists(ROI_result_file):
                print("----------- Warning : submission not ready -----------")
                print("Result not found for ", sub, " and ROI: ", ROI)
                print("Please check if the directory is correct or " + \
                      "generate predicted data for ROI: ", ROI,
                      " in subject: ", sub)
                return
            ROI_result = np.load(ROI_result_file)
            ROI_results[sub] = ROI_result

    submission_pkl = op.join(result_dir, track, track + ".pkl")
    save_dict(results, submission_pkl)

    submission_zip = op.join(result_dir, track, track + ".zip")
    zipped_results = zipfile.ZipFile(submission_zip, 'w')
    zipped_results.write(submission_zip)
    zipped_results.close()


if __name__ == "__main__":
    main()
