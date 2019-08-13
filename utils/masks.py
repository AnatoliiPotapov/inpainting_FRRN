import torch
import numpy as np


def get_mask(img):
    """ Selects (255, 255, 255) pixels as mask """
    # TODO Improve this
    pixel_intensity = np.sum(img, axis=0)
    return (pixel_intensity < 3.0)*1.0

# TODO migrate from np to torch
def create_mask(height=600, width=500, max_masks_count=5):
    mask = np.ones((height, width))

    for _ in range(np.random.randint(1, max_masks_count+1)):
        mask_width = 50 + np.random.randint(0, 30)
        mask_height = 50 + np.random.randint(0, 30)
        
        mask_x = np.random.randint(0, width - mask_width)
        mask_y = np.random.randint(0, height - mask_height)
        
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 0.0
    
    return mask

def get_padding_values(initial_shape=(600,500), factor=8):
    pad_y = (factor - initial_shape[0] % factor) % factor
    pad_x = (factor - initial_shape[1] % factor) % factor
    return pad_y, pad_x


def get_constant_mask(initial_shape=(1,600,500), pad=(0,4)):
    mask = torch.ones(initial_shape)
    pad_zero = torch.nn.ConstantPad2d((0, pad[1], 0, pad[0]), 0)
    return pad_zero(mask)


def get_padding(pad=(0,4), value=0):
    return torch.nn.ConstantPad2d((0, pad[1], 0, pad[0]), 0)

