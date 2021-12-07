import tarfile
from PIL import Image, ImageFile
import os
import os.path
import numpy as np
from numpy.random import randint

import torch.utils.data as data
# ImageFile.LOAD_TRUNCATED_IMAGES = True
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, interval=2, modality='RGB', mode=1, stride=8,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self.interval= interval
        self.mode = mode
        self.stride = stride
        self.t_length = 64
        
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()
        
        
    def _load_image(self, directory, idx):
        img_dir = os.path.join(directory, self.image_tmpl.format(idx))
        img = Image.open(img_dir).convert('RGB')    
        
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [img]
        elif self.modality == 'Flow':
            flow_x, flow_y, _ = img.split()
            x_img = flow_x.convert('L')
            y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        
    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if not (self.mode):  # dense sample
            sample_pos = max(1, 1 + record.num_frames -self.t_length)
            t_stride =self.t_length // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            
            return np.array(offsets) + 1
        else:  # segment sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            
            return offsets + 1        
          

    def _get_val_indices(self, record):
        if (self.mode==0): # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - self.t_length)
            t_stride = self.t_length // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
#                 offsets = np.array([int(tick * x) for x in range(self.num_segments)])                 
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1


    def _get_test_indices(self, record):
        if (self.mode==0):
            sample_pos = max(1, 1 + record.num_frames - self.t_length)
            t_stride = self.t_length // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
#             offsets = np.array([int(tick * x) for x in range(self.num_segments)])            
            return offsets + 1


    def __getitem__(self, index):
        record = self.video_list[index]
        
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        frames, label = self.get(record, segment_indices)
        
        return frames, label, index

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data, _ = self.transform((images,record.label))        
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
