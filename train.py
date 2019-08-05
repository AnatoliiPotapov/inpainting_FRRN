import os 
import yaml 
import random

import numpy as np
import torch
from torch import nn

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

    # build the model and initialize
    inpainting_model = InpaintingModel(config).to(device)
    if checkpoint:
        inpainting_model.load(checkpoint)

    # generator training
    if training:
        print('\nStart training...\n')
        
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

                if i % 100 == 0:
                    print("step:", i, "\tmse:", loss["mse"])
                #if (i+1) % 100 == 0:
                #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                #        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
                keep_training = False

    # generator test
    else:
        print('\nStart testing...\n')
        #generator.test()

    print('Done')
    

# ARGS
if __name__ == "__main__":
    main()