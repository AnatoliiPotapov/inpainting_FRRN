import os 
import glob

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from skimage import io

from utils.model import pad_image


# TODO migrate from np to torch
def create_mask(mask_height, mask_width,
                height=600, width=500, 
                centered=False, max_masks_count=5):
    mask = np.ones((height, width))
    if centered:
        mask_x = int(width/2 - mask_width/2)
        mask_y = int(height/2 - mask_height/2)
        
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 0.0
    else:
        for _ in range(np.random.randint(1, max_masks_count+1)):
            mask_width += np.random.randint(0, 30)
            mask_height += np.random.randint(0, 30)
            
            mask_x = np.random.randint(0, width - mask_width)
            mask_y = np.random.randint(0, height - mask_height)
            
            mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 0.0
        
    return mask


class Dataset(torch.utils.data.Dataset):

    def __init__(self, config, images_path, masks_path=None, training=True):
        super(Dataset, self).__init__()
        
        # loading config 
        self.centered = config['dataset']["centered"]
        self.mask_width = config['dataset']["mask_width"]
        self.mask_height = config['dataset']["mask_height"]
        self.image_width = config['dataset']["image_width"]
        self.image_height = config['dataset']["image_height"]
        self.max_masks_count = config['dataset']["max_masks_count"]
        self.num_workers = config['dataset']["num_workers"]
        self.factor = config['dataset']["factor"]
        
        # loading dataset
        self.flist_path = config['path']["experiment"] + 'file_list.txt'
        if os.path.isfile(self.flist_path):
            print('Loading file list...', self.flist_path)
            self.images = self._read_data_from_file()
        else:
            print('Creating new file list...', self.flist_path)
            self.images = self._load_data(images_path)

        print('Images in dataset:', len(self.images))

        self.masks = None
        if masks_path:
            self.masks = self._load_data(masks_path)
            assert len(self.images) == len(self.masks)
        
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
        image = io.imread(self.images[index])
        image = self._to_tensor(image)
        
        assert image.size()[0] == 3
        
        # crop image
        image = image[:, :self.image_height, :self.image_width]

        if self.masks:
            mask = io.imread(self.masks[index])
        elif not self.training:
            pass # TODO detect mask from image
        else:
            mask = create_mask(self.mask_width, self.mask_height, 
                               width=image.size()[2], height=image.size()[1], 
                               centered=self.centered, max_masks_count=self.max_masks_count)
        
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
        }

    def _load_data(self, data_path):
        files = (
            list(glob.glob(data_path + '**/*.jpg', recursive=True)) + 
            list(glob.glob(data_path + '**/*.png', recursive=True))
        )
        files.sort()
        with open(self.flist_path, 'w') as f:
            for path in files:
                f.write(path + '\n')
        return files

    def _read_data_from_file(self):
        files = None
        with open(self.flist_path, 'r') as f:
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
                #shuffle=True
            )

            for item in image_loader:
                yield item