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

    return pad_zero(image), pad_ones(mask), pad_zero(pad_mask)