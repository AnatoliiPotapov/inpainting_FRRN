import torch
from torch import nn

from .layers import FRRB


class BaseModel(nn.Module):
    """
    """
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        
        self.name = name
        self.config = config


class InpaintingGenerator(nn.Module):
    """
    InpaintingGenerator
    """
    def __init__(self, num_blocks):
        self.frrb = []
        for i in range(num_blocks):
            self.frrb.append(FRRB())
        
    def forward(self, image, mask, pad_mask):
        initial_mask = torch.tensor(mask)
        image = image * initial_mask
        
        for m in self.frrb:
            residuals, mask = m.forward(image, mask)
            residuals *= pad_mask
            image += residuals*(1-initial_mask)
            
        return image, initial_mask  
