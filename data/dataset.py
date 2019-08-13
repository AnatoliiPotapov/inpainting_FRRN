import os 
import glob

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from skimage import io

from utils.model import pad_image
from utils.masks import get_mask, create_mask


class Dataset(torch.utils.data.Dataset):

    def __init__(self, config, training=True):
        super(Dataset, self).__init__()
        
        # loading config 
        self.image_width = config['dataset']["image_width"]
        self.image_height = config['dataset']["image_height"]
        self.max_masks_count = config['dataset']["max_masks_count"]
        self.num_workers = config['dataset']["num_workers"]
        self.factor = config['dataset']["factor"]
        
        # loading dataset
        if training:
            self.dataset_path = config['path']['train']['images']
        else:
            self.dataset_path = config['path']['test']['images']

        print('Loading file list... ', self.dataset_path)
        self.images = self._read_data_from_file()
        
        self.masks = None
        self.training = training
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            item = self._load_item(index)
        except:
            print('loading error: ' + self.images[index])
            item = self._load_item(0)
        return item
        
    def _load_item(self, index):
        image = io.imread(os.path.join(self.dataset_path, self.images[index]))
        image = self._to_tensor(image)
        
        assert image.size()[0] == 3
        
        # crop image
        image = image[:, :self.image_height, :self.image_width]
        
        if not self.training:
            mask = get_mask(image.numpy())
        else:
            mask = create_mask(width=image.size()[2], height=image.size()[1], 
                               max_masks_count=self.max_masks_count)
        
        mask = self._to_tensor(mask)
        
        # padding image
        if self.factor:
            image, mask, constant_mask = pad_image(image, mask, factor=self.factor,
                                                   width=self.image_width, 
                                                   height=self.image_height)
        
        return {
            'image': image,
            'mask': mask,
            'constant_mask': constant_mask,
            'filename': self.images[index],
        }

    def _read_data_from_file(self):
        files = None
        with open(os.path.join(self.dataset_path,'files.txt'), 'r') as f:
            files = f.read().split('\n')[:-1]
        return files

    def _to_tensor(self, image):
        image = Image.fromarray(image)
        image_t = F.to_tensor(image).float()
        return image_t
    
    def create_iterator(self, batch_size):
        while True:
            image_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                # TODO DataLoader worker (pid(s) 17837) exited unexpectedly
                num_workers=self.num_workers,
                drop_last=True,
                shuffle=True if self.training else False
            )

            for item in image_loader:
                yield item

            if not self.training:
                break