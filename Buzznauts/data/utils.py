#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import, division, print_function
import os
import os.path as op
from pathlib import Path
import io
import glob
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import zipfile
import errno
import shutil
from PIL import Image
from decord import VideoReader
from decord import cpu
import numpy as np
import Buzznauts as buzz
from Buzznauts.utils import get_fmri, saveasnii
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Ignore FutureWarnings from nilearn
import warnings
with warnings.catch_warnings():
    # ignore all caught warnings
    warnings.filterwarnings("ignore")
    from nilearn import plotting


url = "https://www.dropbox.com/s/agxyxntrbwko7t1/participants_data.zip?dl=1"
buzz_root = op.dirname(buzz.__path__[0])
data_path = op.join(buzz_root, 'data')
videos_path = op.join(data_path, "stimuli", "videos")
frames_path = op.join(data_path, "stimuli", "frames")


def copyanything(src, dst):
    """Function to copy a source file (or dir) to a destination file (or dir).

    Parameters
    ----------
    'src': str, source file (or dir) path.
    'dst': str, destination path.

    """
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
    if not op.exists(data_dir):
        os.makedirs(data_dir)

    videos_dir = op.join(data_dir, 'stimuli', 'mp4')
    fmri_dir = op.join(data_dir, 'fmri')
    data_url = kwargs.pop('data_url', url)

    if not op.exists(fmri_dir) and not op.exists(videos_dir):
        print("Data downloading...")
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        r = session.get(data_url, allow_redirects=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir)
        print("...Done.")
    else:
        print("Data are already downloaded. Nothing to be done.")
        return

    # Move image files to permanent directory
    tmp_videos_dir = op.join(data_dir, 'AlgonautsVideos268_All_30fpsmax')
    copyanything(tmp_videos_dir, videos_dir)

    tmp_fmri_dir = op.join(data_dir, 'participants_data_v2021')
    copyanything(tmp_fmri_dir, fmri_dir)

    # Clean data directory
    shutil.rmtree(tmp_videos_dir)
    shutil.rmtree(tmp_fmri_dir)

    print(f"Videos are saved at: {videos_dir}")
    print(f"fMRI responses are saved at: {fmri_dir}")

    return


def get_frames_from_mp4(mp4_file, num_frames=None):
    """This function takes a mp4 video file as input and returns a list of
    uniformly sampled frames (PIL Image).

    Parameters
    ----------
    mp4_file : str
        path to mp4 video file
    num_frames : int
        how many frames to select using uniform frame sampling
        Default None, i.e., get all frames

    Returns
    -------
    images : list[PIL Images]
        list of PIL Images
    num_frames : int
        number of frames extracted
    """

    images = list()
    vr = VideoReader(mp4_file, ctx=cpu(0))
    total_frames = len(vr)
    if num_frames is None:
        num_frames = total_frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int)

    for ind in indices:
        images.append( Image.fromarray( vr[ind].asnumpy() ) )

    return images, num_frames

def from_mp4_to_folder(mp4, folder, num_frames=None):
    """This function takes a mp4 video file as input and creates a folder whose
    name is the video name and the contents are the RGB frames of the mp4 video
    saved as an image file with the following naming convention:
        img_00001.jpg, img_00002.jpg ...

    Parameters
    ----------
    mp4 : str
        path to mp4 video file
    folder : str
        root directory to save video folder
    num_frames : int
        how many frames to select using uniform frame sampling
        Default None, i.e., get all frames

    Returns
    -------
    video_folder : str
        path to the folder containing the RGB frames from the mp4 video
    """
    mp4_name = op.splitext(op.basename(mp4))[0]
    video_folder = op.join(folder, mp4_name)
    if not op.exists(video_folder):
        os.makedirs(video_folder)

        images, num_frames = get_frames_from_mp4(mp4, num_frames=num_frames)

        for frame in range(0, num_frames):
            rbg_im = images[frame].convert("RGB")
            im_name = f"img_{frame:05d}.jpg"
            rbg_im.save( op.join(video_folder, im_name) )
    else: # get num_frames
        num_frames = len(glob.glob(video_folder + '/*.jpg'))

    return video_folder, num_frames


def create_videoframe_dataset(videos_src, videoframe_dst):
    """This function converts a folder containing mp4 video files into a video
    dataset with a structure on disk as expected by
    https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch.

    Parameters
    ----------
    videos_src : str)
        root folder containing the video data
    videoframe_dst: str
        Folder that will hold the videoframe dataset

    """
    video_list = glob.glob(videos_src + '/*.mp4')
    video_list.sort()

    if not op.exists(videoframe_dst):
        os.makedirs(videoframe_dst)

    # Each item in the list holds a string with the video metadata in the
    # format: 'PATH START_FRAME END_FRAME LABEL_ID'
    video_metadata = list()
    for mp4 in video_list:
        video_folder, num_frames = from_mp4_to_folder(mp4, videoframe_dst)
        video_name = op.basename(video_folder)
        video_metadata.append(
            f"{video_name} 0 {num_frames-1} {video_name}\n")

    annotations_train = op.join(videoframe_dst, 'annotations_train.txt')
    annotations_test = op.join(videoframe_dst, 'annotations_test.txt')
    annotations = op.join(videoframe_dst, 'annotations.txt')
    create_annotations(video_metadata[:1000], annotations_train)
    create_annotations(video_metadata[1000:], annotations_test)
    create_annotations(video_metadata, annotations)


def create_annotations(videoframe_metadata, annotations_file):
    """This function creates an `annotations.txt` file for a videoframe folder
    containing video data, where each video have its own folder, in which the
    frames of the videos are saved as an RGB image file.

    Parameters
    ----------
    videoframe_metadata : list[str]
        list where each item holds a string with the videoframe metadata in the
        format 'PATH START_FRAME END_FRAME LABEL_ID'
    annotations_file : str
        path to the annotations.txt file that enumerates each video sample

    """
    with open(annotations_file, 'w') as f:
        for metadata in tqdm(videoframe_metadata):
            f.write(metadata)


def plot_video_frames(rows, cols, frame_list, plot_width, plot_height):
    """This function display frames from a video in a grid of axes

    Parameters
    ----------
    rows : int
        Number of rows in the grid of axes
    cols : int
        Number of columns in the grid of axes
    frame_list : list[PIL.Image]
        List of PIL images
    plot_width : float
        Width of the figure
    plot_height : float
        Height of the figure
    """
    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.show()


def visualize_activity_glass(sub, **kwargs):

    # Data parameters
    vid_id = kwargs.pop('video_id', 0)
    fmri_dir = kwargs.pop('fmri_dir', op.join(buzz_root, 'data/fmri'))
    brain_mask = kwargs.pop('brain_mask', op.join(fmri_dir, 'example.nii'))
    track = kwargs.pop('track', 'full_track')
    roi = kwargs.pop('roi', 'WB')

    save_dir = op.join(buzz_root, 'models/baseline/results/alexnet/layer_5')
    save_dir = kwargs.pop('save_dir', save_dir)

    video_id_str = str(vid_id+1).zfill(5)
    nii_save_path = op.join(save_dir, track, sub,
                            f'video_{video_id_str}_{roi}_activity.nii')
    jpg_save_path = op.join(save_dir, track, sub,
                            f'vid_{video_id_str}_{roi}_activity_glass.jpg')
    nii_save_path = kwargs.pop('nii_save_path', nii_save_path)

    nii_folder = Path(nii_save_path).parent.absolute()
    if not op.exists(nii_folder):
        os.makedirs(nii_folder)

    # Plotting parameters
    plot_abs = kwargs.pop('plot_abs', False)
    display_mode = kwargs.pop('display_mode', 'lyr')
    colobar = kwargs.pop('colobar', True)

    # Subject parameters
    score = kwargs.pop('score', None)
    voxel_mask = kwargs.pop('voxel_mask', None)

    visual_mask_3D = np.zeros((78,93,71))
    if score is None:
        track_dir = op.join(fmri_dir, track)
        sub_fmri_dir = op.join(track_dir, sub)
        fmri_data, voxel_mask = get_fmri(sub_fmri_dir, "WB")
        visual_mask_3D[voxel_mask==1] = fmri_data[vid_id, :]
    else:
        visual_mask_3D[voxel_mask==1] = score

    saveasnii(brain_mask, nii_save_path, visual_mask_3D)

    title = f'fMRI response for {sub} watching video {video_id_str}'
    view = plotting.plot_glass_brain(nii_save_path,
                                    title = title,
                                    plot_abs = plot_abs,
                                    display_mode = display_mode,
                                    colorbar = colobar)
    view.savefig(jpg_save_path)
    view.close()
    return jpg_save_path


def visualize_activity_surf(sub, **kwargs):

    # Data parameters
    video_id = kwargs.pop('video_id', 0)
    fmri_dir = kwargs.pop('fmri_dir', op.join(buzz_root, 'data/fmri'))
    brain_mask = kwargs.pop('brain_mask', op.join(fmri_dir, 'example.nii'))
    track = kwargs.pop('track', 'full_track')
    roi = kwargs.pop('roi', 'WB')

    save_dir = op.join(buzz_root, 'models/baseline/results/alexnet/layer_5')
    save_dir = kwargs.pop('save_dir', save_dir)

    nii_save_path = op.join(save_dir, track, sub, f'{roi}_activity.nii')
    nii_save_path = kwargs.pop('nii_save_path', nii_save_path)

    html_save_path = op.join(save_dir, track, sub, f'{roi}_activity_surf.html')
    html_save_path = kwargs.pop('html_save_path', html_save_path)

    # Plotting parameters
    threshold = kwargs.pop('threshold', None)
    surf_mesh = kwargs.pop('surf_mesh', 'fsaverage')
    colobar = kwargs.pop('colobar', True)
    title = f'fMRI response for {sub}'
    title = kwargs.pop('title', title)

    # Subject parameters
    score = kwargs.pop('score', None)
    voxel_mask = kwargs.pop('voxel_mask', None)

    visual_mask_3D = np.zeros((78,93,71))
    if score is None:
        sub_fmri_dir = op.join(track_dir, sub)
        fmri_data, voxel_mask = get_fmri(sub_fmri_dir, "WB")
        visual_mask_3D[voxel_mask==1] = fmri_data[vid_id, :]
    else:
        visual_mask_3D[voxel_mask==1] = score

    saveasnii(brain_mask, nii_save_path, visual_mask_3D)

    view = plotting.view_img_on_surf(nii_save_path,
                                    threshold = threshold,
                                    surf_mesh = surf_mesh,
                                    title = title,
                                    colorbar = colobar)
    view.save_as_html(html_save_path)

    return html_save_path


if __name__ == "__main__":
    download_Algonauts2021()
