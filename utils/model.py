import torch


def pad_image(image, mask, factor=8):
    """
    PAD
    """
    pad_y = (factor - image.shape[-2] % factor) % factor
    pad_x = (factor - image.shape[-1] % factor) % factor
    
    pad_mask = torch.ones(mask.shape)
    pad = torch.nn.ConstantPad2d((0, pad_x, 0, pad_y), 0)

    return pad(image), pad(mask), pad(pad_mask)