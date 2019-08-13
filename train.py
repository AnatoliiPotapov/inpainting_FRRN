import os 
import yaml 
import random

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.general import get_config
from utils.masks import get_constant_mask
from model.net import InpaintingModel
from data.dataset import Dataset
from scripts.metrics import compare_psnr
from utils.progbar import Progbar


os.system("clear")
def main():
    # ARGS 
    config_path = './experiments/local/config.yml'
    masks_path = None
    training = True

    # load config
    code_path = './'
    config, pretty_config = get_config(os.path.join(code_path, config_path))

    print('\nModel configurations:'\
          '\n---------------------------------\n'\
          + pretty_config +\
          '\n---------------------------------\n')

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu"]
    
    # init device
    if config['gpu'] and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        device = torch.device("cpu")

    # initialize random seed
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    if not training:
        np.random.seed(config["seed"])
        random.seed(config["seed"])

    # parse args
    images_path = config['path']['train']
    checkpoint = config['path']['experiment']

    # initialize log writer
    logger = SummaryWriter(log_dir=config['path']['experiment'])

    # build the model and initialize
    inpainting_model = InpaintingModel(config).to(device)
    if checkpoint:
        inpainting_model.load()
    
    # generator training
    if training:
        print('\nStart training...\n')
        batch_size = config['training']['batch_size']

        # create dataset
        dataset = Dataset(config, training=True)
        train_loader = dataset.create_iterator(batch_size)

        test_dataset = Dataset(config, training=False)

        # Train the generator
        total = len(dataset)
        if total == 0:
            raise Exception("Dataset is empty!")

        # Training loop
        epoch = 0
        for i, items in enumerate(train_loader):
            inpainting_model.train()

            if i % total == 0:
                epoch += 1
                print('Epoch', epoch)
                progbar = Progbar(total, width=20, stateful_metrics=['iter'])
                
            images = items['image'].to(device)
            masks = items['mask'].to(device)
            constant_mask = items['constant_mask'].to(device)
            
            # Forward pass
            outputs, residuals, loss, logs = inpainting_model.process(images, masks, constant_mask)
            step = inpainting_model._iteration

            # Backward pass
            inpainting_model.backward(loss)

            # Adding losses to Tensorboard
            for log in logs:
                logger.add_scalar(log[0], log[1], global_step=step)

            if i % config['training']['tf_summary_iters'] == 0:
                grid = torchvision.utils.make_grid(outputs, nrow=4)
                logger.add_image('outputs', grid, step)

                grid = torchvision.utils.make_grid(images, nrow=4)
                logger.add_image('gt', grid, step)
            
            if step % config['training']['save_iters'] == 0:
                # TODO Eval metrics 
                #inpainting_model.save()
                inpainting_model.generator.eval()

                print('Predicting...')
                test_loader = test_dataset.create_iterator(batch_size)
                del images, masks, constant_mask
                
                for i, items in enumerate(test_loader):
                    images = items['image'].to(device)
                    masks = items['mask'].to(device)
                    constant_mask = items['constant_mask'].to(device)
                    result, _, _ = inpainting_model.forward(images, masks, constant_mask)
                    print(result.shape)

            if step >= config['training']['max_iteration']:
                break

            progbar.add(len(images), values=[('iter', step), 
                                             ('loss', loss.cpu().detach().numpy())] + logs)
    # generator test
    else:
        print('\nStart testing...\n')
        #generator.test()

    logger.close()
    print('Done')
    

# ARGS
if __name__ == "__main__":
    main()