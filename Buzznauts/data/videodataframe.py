#!/usr/bin/env python
# coding=utf-8

import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch


class VideoRecord(object):
    """Helper class for class VideoFrameDataset. This class represents a video
    sample's metadata.

    Parameters
    ----------
    root_datapath : str
        the system path to the root folder of the videos.

    row : list[str]
        A list with four or more elements where
        1) The first element is the path to the video sample's frames excluding
        the root_datapath prefix
        2) The second element is the starting frame id of the video
        3) The third element is the inclusive ending frame id of the video
        4) The fourth element is the label index.
        5) any following elements are labels in the case of multi-label
        classification
    """

    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        # +1 because end frame is inclusive
        return self.end_frame - self.start_frame + 1

    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def label(self):
        # just one label_id
        if len(self._data) == 4:
            return self._data[3]
        # sample associated with multiple labels
        else:
            return [label_id for label_id in self._data[3:]]


class VideoFrameDataset(torch.utils.data.Dataset):
    """A highly efficient and adaptable dataset class for videos. Instead of
    loading every frame of a video, loads x RGB frames of a video (sparse
    temporal sampling) and evenly chooses those frames from start to end of the
    video, returning a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x
    WIDTH`` tensors where FRAMES=x if the ``ImglistToTensor()`` transform is
    used.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into
    NUM_SEGMENTS segments and FRAMES_PER_SEGMENT consecutive frames are taken
    from each segment.

    Note
        A demonstration of using this class can be seen in ``demo.py``
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

    Note
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.

    Note
        This class relies on receiving video data in a structure where inside a
        ``ROOT_DATA`` folder, each video lies in its own folder, where each
        video folder contains the frames of the video as individual files with
        a naming convention such as img_001.jpg ... img_059.jpg. For
        enumeration and annotations, this class expects to receive the path to
        a .txt file where each video sample has a row with four (or more in the
        case of multi-label, see README on Github) space separated values:
        ``VIDEO_FOLDER_PATH    START_FRAME    END_FRAME    LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might be
        ``home/data/datasetxyz/videos/``, inside of which a
        ``VIDEO_FOLDER_PATH`` might be ``jumping/0052/`` or ``sample1/`` or
        ``00053/``.

    Parameters
    ----------
    root_path : str
        The root path in which video folders lie. this is ROOT_DATA from the
        description above.

    annotationfile_path : str
        The .txt annotation file containing one row per video sample as
        described above.

    num_segments : int
        The number of segments the video should be divided into to sample
        frames from.

    frames_per_segment : int
        The number of frames that should be loaded per segment. For each
        segment's frame-range, a random start index or the center is chosen,
        from which frames_per_segment consecutive frames are loaded.

    imagefile_template : str
        The image filename template that video frame files have inside of their
        video folders as described above.

    transform : torchvision.transforms.Compose
        Transform pipeline that receives a list of PIL images/frames.

    random_shift : bool
        Whether the frames from each segment should be taken consecutively
        starting from the center of the segment, or  consecutively starting
        from a random location inside the segment range.

    test_mode : bool
        Whether this is a test dataset. If so, chooses frames from segments
        with random_shift=False.
    """

    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 imagefile_template: str = 'img_{:05d}.jpg',
                 transform=None,
                 random_shift: bool = True,
                 test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()

    def _load_image(self, directory, idx):
        im_path = os.path.join(directory, self.imagefile_template.format(idx))
        return [Image.open(im_path).convert('RGB')]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(), self.root_path)
                           for x in open(self.annotationfile_path)]

    def _sample_indices(self, record):
        """For each segment, chooses an index from where frames are to be
        loaded from.

        Parameters
        ----------
        record : VideoRecord
            VideoRecord denoting a video sample.

        Returns
        offsets : list[int]
            List of indices of where the frames of each segment are to be
            loaded from.
        """
        segment_duration = (record.num_frames - self.frames_per_segment + 1)
        segment_duration = segment_duration // self.num_segments

        if segment_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  segment_duration)
            offsets += np.random.randint(segment_duration,
                                         size=self.num_segments)

        # edge cases for when a video has approximately less than
        # (num_frames*frames_per_segment) frames. random sampling in that case,
        # which will lead to repeated frames.
        else:
            offsets = np.sort(np.random.randint(record.num_frames,
                                                size=self.num_segments))

        return offsets

    def _get_val_indices(self, record):
        """For each segment, finds the center frame index.

        Parameters
        ----------
        record : VideoRecord
            VideoRecord denoting a video sample.

        Returns
        -------
        offsets : list[int]
             List of indices of segment center frames.
        """
        if record.num_frames > self.num_segments + self.frames_per_segment - 1:
            offsets = self._get_test_indices(record)

        # edge case for when a video does not have enough frames
        else:
            offsets = np.sort(np.random.randint(record.num_frames,
                                                size=self.num_segments))

        return offsets

    def _get_test_indices(self, record):
        """For each segment, finds the center frame index.

        Parameters
        ----------
            record : VideoRecord
                VideoRecord denoting a video sample

        Returns
        -------
        offsets : list[int]
            List of indices of segment center frames.
        """
        tick = (record.num_frames - self.frames_per_segment + 1)
        tick = tick / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x)
                            for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):
        """For video with id index, loads self.NUM_SEGMENTS *
        self.FRAMES_PER_SEGMENT frames from evenly chosen locations.

        Parameters
        ----------
        index : int
            Video sample index.

        Returns
        -------
        images : list[PIL.Image] or torch.FloatTensor
            a list of PIL images or the result of applying self.transform on
            this list if self.transform is not None.
        """
        record = self.video_list[index]

        if not self.test_mode:
            if self.random_shift:
                segment_indices = self._sample_indices(record)
            else:
                segment_indices = self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self._get(record, segment_indices)

    def _get(self, record, indices):
        """Loads the frames of a video at the corresponding indices.

        Parameters
        ----------
        record : VideoRecord
            VideoRecord denoting a video sample.
        indices : list[int]
            Indices at which to load video frames from.

        Returns
        -------
        images : list[PIL.Image] or list[torch.Tensor]
            A list of PIL images or the result of applying self.transform on
            this list if self.transform is not None.

        label : int
            An integer denoting the video label.
        """
        indices = indices + record.start_frame
        images = list()
        image_indices = list()
        for seg_ind in indices:
            frame_index = int(seg_ind)
            for i in range(self.frames_per_segment):
                seg_img = self._load_image(record.path, frame_index)
                images.extend(seg_img)
                image_indices.append(frame_index)
                if frame_index < record.end_frame:
                    frame_index += 1

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self):
        return len(self.video_list)
    

class FrameDataset(torch.utils.data.Dataset):
    """Frames dataset.
    """
    
    def __init__(self, videoframedataset, transform=None):
        """
        Parameters
        ----------
        videoframedataset : Buzznauts.data.videodataframe.VideoFrameDataset
        transform : torchvision.transforms.transforms.Compose
            callable, optional transform to be applied on a sample
        """
        self.videoframedataset = videoframedataset
        self.transform = transform
        self.num_frames = videoframedataset[0][0].shape[0]
        
    def __len__(self):
        return len(self.videoframedataset)*self.num_frames
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        video_idx, frame_idx = divmod(idx, self.num_frames)
        
        frame = self.videoframedataset[video_idx][0][frame_idx]
        label = self.videoframedataset[video_idx][1]
        
        if self.transform is not None:
            frame = self.transform(frame)
        
        return frame, label


class ImglistToTensor(torch.nn.Module):
    """Converts a list of PIL images in the range [0,255] to a
    torch.FloatTensor of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the
    range [0,1]. Can be used as first transform for ``VideoFrameDataset``.
    """

    def forward(self, img_list):
        """Converts each PIL image in a list to a torch Tensor and stacks them
        into a single tensor.

        Parameters
        ----------
        img_list : list[PIL.Image]
            list of PIL images.

        Returns
        -------
        images : torch.FloatTensor
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        tensor_list = [transforms.functional.to_tensor(pic)
                       for pic in img_list]
        return torch.stack(tensor_list)
