import os 

import torch
from torch import nn

from .layers import InpaintingGenerator
from utils.model import pad_image


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        
        self.name = name
        self.checkpoint = config["architecture"]["checkpoint"]
        self._iteration = 0 #config["iteration"]
    
    # TODO save/load discriminator logic
    def load(self):
        if os.path.exists(self.checkpoint):
            print('Loading %s model...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.checkpoint)
            else:
                data = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
            
            # TODO self.generator
            self.generator.load_state_dict(data['generator'])
            self._iteration = data['iteration']

    def save(self):
        print('Saving %s...\n' % self.name)
        torch.save({
            'iteration': self._iteration,
            # TODO self.generator
            'generator': self.generator.state_dict()
        }, self.checkpoint)


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)
        generator = InpaintingGenerator(config)

        gpus = [int(i) for i in config["gpu"].split(",")]
        if len(gpus) > 1:
            # TODO different gpus ids
            gpus = list(range(len(gpus)))
            generator = nn.DataParallel(generator, gpus)
        self.add_module('generator', generator)
        
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()
        self.add_module('l1_loss', l1_loss)
        self.add_module('mse_loss', mse_loss)
        
        learning_rate = config['training']["learning_rate"]
        betas = (config['training']["beta1"], config['training']["beta2"])
        self.optimizer = torch.optim.Adam(generator.parameters(), 
                                     lr=learning_rate, betas=betas)

    def process(self, images, masks, pad_image):
        self._iteration += 1
        copy_images = images.clone().detach().requires_grad_(False)

        # zero optimizers
        self.optimizer.zero_grad()

        # process outputs
        outputs, residuals, initial_mask = self(images, masks, pad_image)

        losses = {
            "l1": self.l1_loss(outputs, copy_images),
            "mse": self.mse_loss(outputs, copy_images),
        }

        return outputs, residuals, losses

    def forward(self, images, masks, pad_masks):
        return self.generator(images, masks, pad_masks)

    def backward(self, l1_loss=None, mse_loss=None):
        if l1_loss is not None:
            l1_loss.backward()
        self.optimizer.step()

        if mse_loss is not None:
            mse_loss.backward()
        self.optimizer.step()