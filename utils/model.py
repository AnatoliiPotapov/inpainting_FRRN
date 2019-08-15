import numpy as np
import torch


def pad_image(image, mask, width, height, factor=8):
    """
    PAD
    """
    image_h, image_w = image.size()[1:]

    pad_y = height - image_h
    pad_x = width - image_w
    
    pad_y += (factor - height % factor) % factor
    pad_x += (factor - width % factor) % factor

    pad_mask = torch.ones(mask.shape, device=mask.device)
    
    pad_zero = torch.nn.ConstantPad2d((0, pad_x, 0, pad_y), 0)
    pad_ones = torch.nn.ConstantPad2d((0, pad_x, 0, pad_y), 1)

    return pad_zero(image), pad_zero(mask), pad_zero(pad_mask)


def random_crop(images, masks, constant_mask, strip_size):
    min_dim = min(images.size()[2:])

    if np.random.randint(0, 2):
        new_h, new_w = strip_size, min_dim
    else:
        new_h, new_w = min_dim, strip_size

    top = np.random.randint(0, min_dim - new_h) if new_h != min_dim else 0
    left = np.random.randint(0, min_dim - new_w) if new_w != min_dim else 0

    images = images[:, :, 
                    top: top + new_h, 
                    left: left + new_w]

    masks = masks[:, :, 
                  top: top + new_h, 
                  left: left + new_w]

    constant_mask = constant_mask[:, :, 
                                  top: top + new_h, 
                                  left: left + new_w]

    return images, masks, constant_mask