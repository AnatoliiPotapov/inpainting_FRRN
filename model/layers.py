import math
from pytorch_memlab import profile

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from .pconv import PartialConv2d
from .config import get_model


class PConvBlock(nn.Module):
    """ PConvBlock includes:
    
          Upsampling (optional) 
          PartialConv2d
          Activation Function
    """
    def __init__(self, in_channels, out_channels, stride=1, 
                 kernel_size=3, upscale=False, act='relu'):
        super().__init__()
        
        self.upscale = None
        if upscale:
            self.upscale = nn.Upsample(scale_factor=2, mode='nearest')

        pad_size = math.trunc(kernel_size / 2)
        self.pconv = PartialConv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=pad_size, return_mask=True)
        self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        
        if act == 'relu':
            self.act = nn.ReLU(False)
        elif act == 'leaky':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act =  nn.Identity()
        
    def forward(self, x, mask):
        if self.upscale:
            x = self.upscale(x)
            mask = self.upscale(mask)
        x, mask = self.pconv(x, mask_in=mask)
        return self.act(self.norm(x)), mask


class Stack(nn.Module):
    """
    Chainer for FRRB blocks
    """
    def __init__(self, *modules):
        super().__init__()
        self.mod = nn.ModuleList(modules)
        
    def forward(self, x, mask):
        for module in self.mod:
            x, mask = module(x, mask)
        return x, mask


class FRRB(nn.Module):
    """
    Full Resolution Residual Block
    """
    def __init__(self, config):
        super().__init__()
        self.conf = get_model(config['architecture']['model'])
        self.left = self.get_pipeline_from_config(self.conf['left'])
        self.right = self.get_pipeline_from_config(self.conf['right'])

    def forward(self, x, mask, constant_mask):
        
        left_r, left_mask = self.left(x, mask)
        right_r, right_mask = self.right(x, mask)
    
        mask = left_mask * right_mask 
        if constant_mask is not None:
            mask *= constant_mask
        r = 0.5 * ((left_r + right_r) * mask).float()
        
        return r, mask   

    def get_pipeline_from_config(self, conf):
        pipe = []
        for i in range(len(conf)):
            in_channels = 3 if i==0 else conf[i-1][2]
            out_channels = conf[i][2]
            pipe.append(
                PConvBlock(in_channels, out_channels, stride=conf[i][0],
                           kernel_size=conf[i][1],
                           upscale=True if conf[i][3]=='u' else False,
                           act=conf[i][4])
            )
        return Stack(*pipe)


class InpaintingGenerator(nn.Module):
    """
    InpaintingGenerator
    """
    def __init__(self, config):
        super(InpaintingGenerator, self).__init__()
        self.frrb_1 = nn.ModuleList([
            FRRB(config) for i in range(config["architecture"]["num_blocks"])
        ])
        self.frrb_2 = nn.ModuleList([
            FRRB(config) for i in range(config["architecture"]["num_blocks"])
        ])

    def forward(self, image, mask, constant_mask, alpha):
        initial_mask = mask.clone().detach()
        result_gt = image.clone()
        result = image.clone() * mask

        residuals = []
        res_masks = []
        for f_1, f_2 in zip(self.frrb_1, self.frrb_2):
            result = alpha*result_gt + (1-alpha)*result

            residual_1, _ = f_1(result, mask, constant_mask)
            result_1 = result + residual_1 * (1 - initial_mask)
            residual_2, mask = f_2(result_1, mask, constant_mask)
            result = result_1 + residual_2 * (1 - initial_mask)
            result = result.clamp(min=0.0, max=1.0)

            res_masks.append(mask - initial_mask)
            residuals.append(result)

        return result, residuals, res_masks

    def predict(self, image, mask, constant_mask):
        initial_mask = mask.clone().detach()
        result = image.clone() * mask

        with torch.no_grad():
            for f_1, f_2 in zip(self.frrb_1, self.frrb_2):
                residual_1, _ = f_1(result, mask, constant_mask)
                del _
                result_1 = result + residual_1 * (1 - initial_mask)
                residual_2, mask = f_2(result_1, mask, constant_mask)
                del f_1, f_2
                result = result_1 + residual_2 * (1 - initial_mask)
                result = result.clamp(min=0.0, max=1.0)

        return result
    

class InpaintingDiscriminator(nn.Module):
    """
    InpaintingDiscriminator
    """
    def __init__(self, config):
        super(InpaintingDiscriminator, self).__init__()

        self.use_sigmoid = True

        self.arch = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        outputs = self.arch(x)
        if self.use_sigmoid:
            outputs = torch.sigmoid(outputs)

        return outputs