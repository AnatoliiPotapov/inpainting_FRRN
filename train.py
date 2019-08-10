import os 
import yaml 
import random

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model.net import InpaintingModel
from data.dataset import Dataset


# ARGS 
config_path = 'experiments/config.yml'
images_path = '../DATASET_INPAINTING/temp_gt/'
masks_path = None#'../DATASET_INPAINTING/images'
checkpoint = None
training = True


def plot_image(tensor, exit=False):
    import matplotlib.pyplot as plt 
    plt.imshow(tensor.permute(1,2,0).detach().numpy())
    plt.show()
    if exit:
        exit()

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
    inpainting_model = InpaintingModel(config).to(device)
    if checkpoint:
        inpainting_model.load(checkpoint)

    # generator training
    if training:
        print('\nStart training...\n')
        inpainting_model.train()
        batch_size = config['training']['batch_size']

        # create dataset
        dataset = Dataset(config['dataset'], images_path, masks_path, training)
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
                padded_images = items['padded_image'].to(device)
                
                images = (images/255).float()
                
                # Forward pass
                # TODO inpainting_model.train() with losses
                outputs, loss = inpainting_model.process(images, masks, padded_images)
                inpainting_model.backward(mse_loss=loss['mse'])
                step = inpainting_model._iteration

                #logger.add_scalar('loss_l1', loss['l1'].item(), global_step=step)
                logger.add_scalar('loss_mse', loss['mse'].item(), global_step=step)

                if i % 100 == 0:
                    print("step:", i, "\tmse:", loss["mse"].item())
                    
                    #images = outputs.detach() * 255
                    grid = torchvision.utils.make_grid(images*255, nrow=4)
                    logger.add_image('gt_images', grid, step)

                    #images = outputs.detach() * 255
                    grid = torchvision.utils.make_grid(outputs.detach()*255, nrow=4)
                    logger.add_image('outputs', grid, step)

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