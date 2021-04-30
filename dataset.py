import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import cv2
from PIL import Image
import numpy as np

import pandas as pd
from skimage import io, transform

class ToolDataset(Dataset):
    """Tool Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1,2)

        
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]
        #landmarks = landmarks.astype(int)

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
    
def generate_heatmap(heatmap, pt, sigma):

    heatmap[int(pt[1])][int(pt[0])] = 1
        
    
    
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    am = np.amax(heatmap)
    heatmap /= am
    return heatmap

class LocsToHeatmaps:

    def __init__(self, img_size=(256,256), out_size=(64,64), sigma=1):
        self.img_size = img_size
        self.out_size = out_size
        self.x_scale = 1.0 * out_size[0]/img_size[0]
        self.y_scale = 1.0 * out_size[1]/img_size[1]
        self.sigma=sigma
        x = np.arange(0, out_size[1], dtype=np.float)
        y = np.arange(0, out_size[0], dtype=np.float)
        self.yg, self.xg = np.meshgrid(y,x, indexing='ij')
        return

    def __call__(self, sample):
        landmarks = sample['landmarks']
        sample['keypoints'] = np.concatenate((landmarks, np.ones((4,1))), axis=-1)
        sigma = 7
        gaussian_hm = np.zeros((self.out_size[0], self.out_size[1], sample['keypoints'].shape[0]))
        for i,keypoint in enumerate(sample['keypoints']):
            if keypoint[2] != 0:
                gaussian_hm[:,:,i] = generate_heatmap(gaussian_hm[:,:,i], tuple(keypoint[:-1].astype(np.int) * self.x_scale), (sigma, sigma))
        sample['keypoint_locs'] = sample['keypoints'][:,:2]
        sample['visible_keypoints'] = sample['keypoints'][:,2]
        sample['keypoint_heatmaps'] = gaussian_hm
        return sample
    
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks,keypoint_heatmaps = sample['image'], sample['landmarks'], sample['keypoint_heatmaps']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.reshape(-1)
        if 'keypoint_heatmaps' in sample:
            
            keypoint_heatmaps =\
                torch.from_numpy(keypoint_heatmaps.astype(np.float32).transpose(2,0,1))
            
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks),
                'keypoint_heatmaps': keypoint_heatmaps
               }