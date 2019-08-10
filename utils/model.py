import torch


def pad_image(image, mask, factor=8):
    """
    PAD
    """
    pad_y = (factor - image.shape[-2] % factor) % factor
    pad_x = (factor - image.shape[-1] % factor) % factor
    
    pad_mask = torch.ones(mask.shape, device=mask.device)
    
    pad_zero = torch.nn.ConstantPad2d((0, pad_x, 0, pad_y), 0)
    pad_ones = torch.nn.ConstantPad2d((0, pad_x, 0, pad_y), 1)

    return pad_zero(image), pad_ones(mask), pad_zero(pad_mask)