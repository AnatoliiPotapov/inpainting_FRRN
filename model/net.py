import os 

import torch
from torch import nn

from model.layers import InpaintingGenerator
from model.utils import pad_image
from model.loss import AdversarialLoss, StyleLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        
        self.name = name
        self.checkpoint = config["path"]["experiment"]
        self.checkpoint += self.name + '.ckpt'
        self._iteration = 0 #config["iteration"]
    
    # TODO save/load discriminator logic
    def load(self):
        if os.path.isfile(self.checkpoint):
            print('Loading %s model...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.checkpoint)
            else:
                data = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
            
            # TODO self.generator
            self.generator.load_state_dict(data['generator'])
            self._iteration = data['iteration']
        else:
            print('Checkpoint %s not found!' % self.checkpoint)

    def save(self):
        print('\nSaving %s...\n' % self.name)
        torch.save({
            'iteration': self._iteration,
            # TODO self.generator
            'generator': self.generator.state_dict()
        }, self.checkpoint)


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)
        generator = InpaintingGenerator(config)

        if config["gpu"]:
            gpus = [int(i) for i in config["gpu"].split(",")]
    
            if len(gpus) > 1:
                # TODO different gpus ids
                gpus = list(range(len(gpus)))
                generator = nn.DataParallel(generator, gpus)
        self.add_module('generator', generator)
        
        l1_loss = nn.L1Loss()
        self.add_module('l1_loss', l1_loss)
        self.rec_loss_weight = config['training']["rec_loss_weight"]

        mse_loss = nn.MSELoss()
        self.add_module('mse_loss', mse_loss)
        self.mse_loss_weight = config['training']["mse_loss_weight"]

        style_loss = StyleLoss()
        self.add_module('style_loss', style_loss)
        self.style_loss_weight = config['training']["style_loss_weight"]

        
        learning_rate = config['training']["learning_rate"]
        betas = (config['training']["beta1"], config['training']["beta2"])
        self.optimizer = torch.optim.Adam(generator.parameters(), 
                                     lr=learning_rate, betas=betas)

    def process(self, images, masks, pad_image):
        self._iteration += 1

        # zero optimizers
        self.optimizer.zero_grad()

        images_gt = images.clone()

        # process outputs
        outputs = self(images_gt, masks, pad_image)
        
        # losses 
        mse_loss = self.mse_loss(outputs, images)
        mse_loss *= self.mse_loss_weight
        style_loss = self.style_loss(outputs * (1-masks), images * (1-masks))
        style_loss *= self.style_loss_weight
        rec_loss = self.l1_loss(outputs * (1-masks), images * (1-masks))
        rec_loss *= self.rec_loss_weight
        # TODO Step loss
         
        loss = style_loss + mse_loss + rec_loss
        logs = [
            ("mse", mse_loss.item()),
            ("style", style_loss.item()),
            ("rec", rec_loss.item()),
        ]
        
        return outputs, loss, logs

    def forward(self, images, masks, pad_masks):
        return self.generator(images, masks, pad_masks)

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()