#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import, division, print_function
import os
import os.path as op
import io
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import zipfile
import errno
import shutil

url = "https://www.dropbox.com/s/agxyxntrbwko7t1/participants_data.zip?dl=1"
data_path = "../Buzznauts/data"

def copyanything(src, dst):
        try:
            shutil.copytree(src, dst)
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                shutil.copy(src, dst)
            else: raise

def download_Algonauts2021(**kwargs):
    """Function to download data from the Algonauts Challenge 2021.

    Parameters
    ----------
    kwargs: dict
    'data_dir': str, data directory path. Default: ../Buzznauts/data
    'data_url'    : str, data url. Default:
        https://www.dropbox.com/s/agxyxntrbwko7t1/participants_data.zip?dl=1

    """
    data_dir = kwargs.pop('data_dir', data_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    videos_dir = op.join(data_dir, 'videos')
    fmri_dir = op.join(data_dir, 'fmri')
    data_url = kwargs.pop('data_url', url)

    if not os.path.exists(fmri_dir) and not os.path.exists(videos_dir):
        print("...Data downloading...")
        r = requests.get(data_url, allow_redirects=True)
        # session = requests.Session()
        # retry = Retry(connect=3, backoff_factor=0.5)
        # adapter = HTTPAdapter(max_retries=retry)
        # session.mount('http://', adapter)
        # session.mount('https://', adapter)
        # r = session.get(data_url, allow_redirects=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir)
        print("...Done.")
    else:
        print("Data are already downloaded. Nothing to be done.")
        return

    # Move image files to permanent directory
    tmp_videos_dir = op.join(data_dir, 'AlgonautsVideos268_All_30fpsmax')
    copyanything(op.join(tmp_videos_dir, videos_dir)

    tmp_fmri_dir = op.join(data_dir, 'participants_data_v2021')
    copyanything(op.join(tmp_fmri_dir), fmri_dir)

    # Clean data directory
    shutil.rmtree(tmp_videos_dir)
    shutil.rmtree(tmp_fmri_dir)

    print(f"Videos are saved at: {videos_dir}")
    print(f"fMRI responses are saved at: {fmri_dir}")

    return

if __name__ == "__main__":
    download_Algonauts2021()
