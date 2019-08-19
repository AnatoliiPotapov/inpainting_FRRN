import torch
import numpy as np


def get_mask(img):
    """ Selects (255, 255, 255) pixels as mask """
    # TODO Improve this
    pixel_intensity = np.sum(img, axis=0)
    return (pixel_intensity < 3.0)*1.0

# TODO migrate from np to torch
def create_mask(height=600, width=500, max_masks_count=5, init_mask=None):
    mask = np.zeros((height, width))

    for _ in range(np.random.randint(1, max_masks_count+1)):
        
        if init_mask is not None:
            old_mask = mask.copy()
            for count in range(10):
                mask_width = 80 + np.random.randint(0, 40)
                mask_height = 80 + np.random.randint(0, 40)

                mask_x = np.random.randint(-20, width - mask_width + 20)
                mask_y = np.random.randint(-20, height - mask_height + 20)

                mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1.0
                if (init_mask + mask).max() == 255:
                    break
                mask = old_mask.copy()
    
        else:
            mask_width = 80 + np.random.randint(0, 40)
            mask_height = 80 + np.random.randint(0, 40)

            mask_x = np.random.randint(-20, width - mask_width + 20)
            mask_y = np.random.randint(-20, height - mask_height + 20)

            mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1.0
    
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

