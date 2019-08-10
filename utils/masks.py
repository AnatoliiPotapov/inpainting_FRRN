import torch


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

