import os 

import torch
from torch import nn

from model.layers import InpaintingGenerator, InpaintingDiscriminator
from model.loss import AdversarialLoss, StyleLoss, PerceptualLoss
from utils.model import random_alpha
from .optimizer import RAdam


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        
        self.name = name
        self.checkpoint = config["path"]["experiment"]
        self._iteration = 0

        self.with_discriminator = config['training']['discriminator']
    
    def load(self):
        iterations = []
        for f in os.listdir(self.checkpoint):
            if '.ckpt' in f:
                iterations.append(int(f.split('-')[-1]))
        
        if iterations:
            checkpoint = self.checkpoint + self.name + '.ckpt-' + str(max(iterations))
            print('Loading generator %s...' % checkpoint)

            if torch.cuda.is_available():
                data = torch.load(checkpoint)
            else:
                data = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            
            self.generator.load_state_dict(data['generator'], strict=False)
            self._iteration = data['iteration']

            if self.with_discriminator:
                iterations = []
                for f in os.listdir(self.checkpoint):
                    if '.ckpt' in f and '_dis' in f:
                        iterations.append(int(f.split('-')[-1]))
                
                if len(iterations)>0:
                    checkpoint = self.checkpoint + self.name + '_dis' + '.ckpt-' + str(max(iterations))
                    print('Loading discriminator %s...' % checkpoint)
                    
                    if torch.cuda.is_available():
                        data = torch.load(checkpoint)
                    else:
                        data = torch.load(checkpoint, map_location=lambda storage, loc: storage)
                    
                    self.discriminator.load_state_dict(data['discriminator'], strict=False)
                else:
                    print('No checkpoints for discriminator...')
        else:
            print('Checkpoint', self.checkpoint + self.name + '.ckpt-{iter}', 'not found!')

    def save(self):
        checkpoint = self.checkpoint + self.name + '.ckpt-' + str(self._iteration)
        print('\nSaving generator %s...\n' % checkpoint)
        torch.save({
            'iteration': self._iteration,
            'generator': self.generator.state_dict()
        }, checkpoint)

        if self.with_discriminator:
            checkpoint = self.checkpoint + self.name + '_dis' + '.ckpt-' + str(self._iteration)
            print('Saving discriminator %s...\n' % checkpoint)
            torch.save({
                'discriminator': self.discriminator.state_dict()
            }, checkpoint)

class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)
        generator = InpaintingGenerator(config)
        self.with_discriminator = config['training']['discriminator']
        if self.with_discriminator:
            discriminator = InpaintingDiscriminator(config)

        if config["gpu"]:
            gpus = [int(i) for i in config["gpu"].split(",")]
    
            if len(gpus) > 1:
                gpus = list(range(len(gpus)))
                generator = nn.DataParallel(generator, gpus)
                if self.with_discriminator:
                    discriminator = nn.DataParallel(discriminator, gpus)

        self.add_module('generator', generator)

        l1_loss = nn.L1Loss()
        self.add_module('l1_loss', l1_loss)
        self.rec_loss_weight = config['training']["rec_loss_weight"]
        self.step_loss_weight = config['training']["step_loss_weight"]

        mse_loss = nn.MSELoss()
        self.add_module('mse_loss', mse_loss)
        self.mse_loss_weight = config['training']["mse_loss_weight"]

        style_loss = StyleLoss()
        self.add_module('style_loss', style_loss)
        self.style_loss_weight = config['training']["style_loss_weight"]
        
        per_loss = PerceptualLoss()
        self.add_module('per_loss', per_loss)
        self.per_loss_weight = config['training']["per_loss_weight"]

        learning_rate = config['training']["learning_rate"]
        betas = (config['training']["beta1"], config['training']["beta2"])
        
        if config['training']['optimizer'] == 'adam':
            self.gen_optimizer = torch.optim.Adam(generator.parameters(), 
                                        lr=learning_rate, betas=betas)
        elif config['training']['optimizer'] == 'radam':
            self.gen_optimizer = RAdam(generator.parameters(), 
                                    lr=learning_rate, betas=betas)

        if self.with_discriminator:
            self.add_module('discriminator', discriminator)
            adversarial_loss = AdversarialLoss(type=config['training']['gan_loss'])
            self.add_module('adversarial_loss', adversarial_loss)
            self.adversarial_loss_weight = config['training']["adv_loss_weight"]

            self.dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                        lr=learning_rate * config['training']['d2g_lr'],
                        betas=betas)

        # Teacher forcing
        self.beta = config['training']['beta']

        self.alpha = config['training']['alpha']
        self.alpha_decay = config['training']['alpha_decay']
        self.alpha_decay_start_iter = config['training']['alpha_decay_start_iter']

    def process(self, images, masks, constant_mask):
        self._iteration += 1
        images_gt = images.clone().detach().requires_grad_(False)

        # zero optimizers
        self.gen_optimizer.zero_grad()

        # process outputs
        outputs, residuals, res_masks = self(images, masks, constant_mask)

        if self.with_discriminator:
            self.dis_optimizer.zero_grad()

            # discriminator loss
            dis_input_real = images_gt
            dis_input_fake = outputs.detach()
            dis_real = self.discriminator(dis_input_real)                    # in: [rgb(3)]
            dis_fake = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_adv_loss = (dis_real_loss + dis_fake_loss) / 2

            # generator adversarial loss
            gen_input_fake = outputs
            gen_fake = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
            gen_adv_loss = self.adversarial_loss(gen_fake, True, False)
            gen_adv_loss *= self.adversarial_loss_weight

        # alpha
        if self.alpha_decay_start_iter > 0:
            self.alpha_decay_start_iter -= 1
        elif self.alpha > 0:
            self.alpha -= self.alpha_decay
            self.alpha = max(self.alpha, 0.0)

        # losses 
        mse_loss = self.mse_loss(outputs, images_gt)
        mse_loss *= self.mse_loss_weight
        
        style_loss = self.style_loss(outputs * (1-masks), images_gt * (1-masks))
        style_loss *= self.style_loss_weight
        
        rec_loss = self.l1_loss(outputs * (1-masks), images_gt * (1-masks))
        rec_loss *= self.rec_loss_weight

        per_loss = self.perceptual_loss(outputs, images_gt)
        per_loss *= self.per_loss_weight
        
        step_loss = 0
        for r, m in zip(residuals, res_masks):
            step_loss += torch.mean(torch.abs((r-images_gt)*m))
        step_loss /= len(residuals)
        step_loss *= self.step_loss_weight

        loss = style_loss + mse_loss + rec_loss + step_loss + per_loss
        logs = [
            ("mse", mse_loss.item()),
            ("style", style_loss.item()),
            ("rec", rec_loss.item()),
            ("step", step_loss.item()),
            ("per", per_loss.item()),
            ("alpha", self.alpha),
        ]

        if self.with_discriminator:
            gen_loss = loss + gen_adv_loss
            logs += [
                ("gen_adv_loss", gen_adv_loss.item()),
                ("dis_adv_loss", dis_adv_loss.item()),
            ] 
            return outputs, residuals, gen_loss, dis_adv_loss, logs

        return outputs, residuals, loss, logs

    def forward(self, images, masks, constant_mask):
        if self.alpha > 0:
            alpha = random_alpha(self.alpha, self.beta)
            assert alpha <= self.alpha
        else:
            alpha = 0

        return self.generator(images, masks, constant_mask, alpha)

    def backward(self, gen_loss, dis_loss=None):
        gen_loss.backward()
        self.gen_optimizer.step()
        
        if self.with_discriminator:
            dis_loss.backward()
            self.dis_optimizer.step()