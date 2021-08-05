#!/usr/bin/env python
# coding=utf-8

url = " https://www.dropbox.com/s/agxyxntrbwko7t1/participants_data.zip?dl=1"

import os
import io
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import zipfile
import shutil

def download_Algonauts2021(**kwargs):
    """Function to download data from the Algonauts Challenge 2021.

    Parameters
    ----------
    kwargs: dict
    'data_dir': str, data directory path. Default: ./data
    'data_url'    : str, data url. Default:
        https://www.dropbox.com/s/agxyxntrbwko7t1/participants_data.zip?dl=1

    """
    cwd = os.getcwd()
    data_dir = kwargs.pop('data_dir', os.path.join(cwd, "data"))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    videos_dir = os.path.join(data_dir, 'videos')
    fmri_dir = os.path.join(data_dir, 'fmri')
    data_url = kwargs.pop('data_url', url)

    if not os.path.exists(fmri_dir) and not os.path.exists(videos_dir):
        print("...Data downloading...")
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        r = session.get(data_url, allow_redirects=True, verify=False)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        print("...Done.")
    else:
        print("Data are already downloaded. Nothing to be done.")
        return

    # Move image files to permanent directory
    tmp_videos_dir = os.path.join(data_dir, 'AlgonautsVideos268_All_30fpsmax')
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
        for f in os.listdir(tmp_videos_dir):
            shutil.copy(os.path.join(tmp_videos_dir, f), videos_dir)

    tmp_fmri_dir = os.path.join(data_dir, 'participants_data_v2021')
    if not os.path.exists(fmri_dir):
        os.makedirs(fmri_dir)
        for f in os.listdir(tmp_fmri_dir):
            shutil.copy(os.path.join(tmp_fmri_dir, f), fmri_dir)

    # Clean data directory
    shutil.rmtree(tmp_videos_dir)
    shutil.rmtree(tmp_fmri_dir)

    print(f"Videos are saved at: {videos_dir}")
    print(f"fMRI responses are saved at: {fmri_dir}")

    return

if __name__ == "__main__":
    download_Algonauts2021()
