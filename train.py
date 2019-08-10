import os 
import yaml 
import random

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.masks import get_constant_mask
from model.net import InpaintingModel
from data.dataset import Dataset


# ARGS 
config_path = 'experiments/config.yml'
images_path = '../Datasets/Huawei/DATASET_INPAINTING/train_gt/'
masks_path = None#'../DATASET_INPAINTING/images'
checkpoint = None
training = True


os.system("clear")
def main():
    # load config
    code_path = './'
    with open(os.path.join(code_path, config_path), 'r') as f:
        pretty_config = f.read()
        config = yaml.load(pretty_config, yaml.Loader)

    print('\nModel configurations:'\
          '\n---------------------------------\n'\
          + pretty_config +\
          '\n---------------------------------\n')

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu"]
    
    # init device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        device = torch.device("cpu")

    # initialize random seed
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # initialize log writer
    logger = SummaryWriter(log_dir=config['path']['experiment'])

    # build the model and initialize
    initial_mask = get_constant_mask().to(device)
    inpainting_model = InpaintingModel(config, initial_mask).to(device)
    if checkpoint:
        inpainting_model.load(checkpoint)

    # generator training
    if training:
        print('\nStart training...\n')
        inpainting_model.train()
        batch_size = config['training']['batch_size']

        # create dataset
        dataset = Dataset(config['dataset'], config['path']['train'], masks_path, training)
        train_loader = dataset.create_iterator(batch_size)

        # Train the generator
        keep_training = True
        total = len(dataset)
        if total == 0:
            raise Exception("Dataset is empty!")

        while keep_training:
            #progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for i, items in enumerate(train_loader):
                images = items['image'].to(device)
                masks = items['mask'].to(device)
                
                # Forward pass
                outputs, residuals, loss = inpainting_model.process(images, masks)
                step = inpainting_model._iteration

                logger.add_scalar('loss_l1', loss['l1'].item(), global_step=step)
                logger.add_scalar('loss_mse', loss['mse'].item(), global_step=step)

                inpainting_model.backward(loss['mse'])

                if i % 100 == 0:
                    print("step:", i, "\tmse:", loss["mse"].item())
                    
                    #images = outputs.detach() * 255
                    grid = torchvision.utils.make_grid(images*255, nrow=4)
                    logger.add_image('gt_images', grid, step)

                    #images = outputs.detach() * 255
                    grid = torchvision.utils.make_grid(outputs.detach()*255, nrow=4)
                    logger.add_image('outputs', grid, step)

                    #images = outputs.detach() * 255
                    grid = torchvision.utils.make_grid((residuals.detach()*(1-masks.detach()))*255, nrow=4)
                    logger.add_image('residuals', grid, step)



                keep_training = False

    # generator test
    else:
        print('\nStart testing...\n')
        #generator.test()

    logger.close()
    print('Done')
    

# ARGS
if __name__ == "__main__":
    main()