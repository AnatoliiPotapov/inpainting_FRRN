import math
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from .pconv import PartialConv2d

class PConvBlock(nn.Module):
    """ PConvBlock includes:
    
          Upsampling (optional) 
          PartialConv2d
          Activation Function
    """
    def __init__(self, in_channels, out_channels, stride=1, 
                 kernel_size=3, upscale=False):
        super().__init__()
        
        self.upscale = None
        if upscale:
            self.upscale = nn.Upsample(scale_factor=2, mode='nearest')

        pad_size = math.trunc(kernel_size / 2)
        self.pconv = PartialConv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=pad_size, return_mask=True)
        self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        self.act = nn.ReLU(False)    
        
    def forward(self, x, mask):
        if self.upscale:
            x = self.upscale(x)
            mask = self.upscale(mask)
        x, mask = self.pconv(x, mask_in=mask)
        return self.act(self.norm(x)), mask


class Stack(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        
    def forward(self, x, mask):
        for module in self.modules:
            x, mask = module.forward(x, mask)
        return x, mask


class FRRB(nn.Module):
    def __init__(self, constant_mask=None):
        super().__init__()
        self.conf = {
            'left': [(2,3,64,'r'), (2,3,96,'r'), (2,3,128,'r'),
                     (1,3,96,'u'), (1,3,64,'u'), (1,3,3,'u')],
            'right': [(1,5,32,'r'), (1,5,32,'r'), (1,5,3,'r')]
        }
        self.constant_mask = constant_mask
        self.left = self.get_pipeline_from_config(self.conf['left'])
        self.right = self.get_pipeline_from_config(self.conf['right'])

    def forward(self, x, mask):
        left_r, left_mask = self.left(x, mask)
        right_r, right_mask = self.right(x, mask)
        mask = left_mask * right_mask 
        if self.constant_mask is not None:
            mask *= self.constant_mask
        r = 0.5 * (left_r + right_r) * mask
        return r, mask   

    def get_pipeline_from_config(self, conf):
        pipe = []
        for i in range(len(conf)):
            in_channels = 3 if i==0 else conf[i-1][2]
            out_channels = conf[i][2]
            pipe.append(
                PConvBlock(in_channels, out_channels, stride=conf[i][0],
                           kernel_size=conf[i][1],
                           upscale=True if conf[i][3]=='u' else False)
            )
        return Stack(*pipe)