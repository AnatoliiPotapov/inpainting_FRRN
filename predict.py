import os 
import yaml 
import random
import argparse

import numpy as np

from utils.general import get_config
from utils.progbar import Progbar
from model.net import InpaintingModel
from data.dataset import Dataset
from scripts.metrics import compute_metrics


os.system("clear")
def main(images_path, pred_path, 
         config_path, experiment_path, 
         batch_size):
    # load config
    code_path = './'
    config, pretty_config = get_config(os.path.join(code_path, config_path))
    config['experiment'] = os.path.join(experiment_path, config['experiment'])

    print('\nModel configurations:'\
          '\n---------------------------------\n'\
          + pretty_config +\
          '\n---------------------------------\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']

    # Import Torch after os env
    import torch
    import torchvision
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.utils import save_image

    # init device
    if config['gpu'] and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        device = torch.device("cpu")

    # initialize random seed
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # parse args
    checkpoint = config['path']['experiment']

    # build the model and initialize
    inpainting_model = InpaintingModel(config).to(device)
    if checkpoint:
        inpainting_model.load()
    inpainting_model.alpha = 0.0
    
    pred_directory = os.path.join(checkpoint, 'predictions')
    if not os.path.exists(pred_directory):
        os.makedirs(pred_directory)

    # predict 
    dataset = Dataset(config, training=False)

    # Train the generator
    total = len(dataset)
    if total == 0:
        raise Exception("Dataset is empty!")

    print('Predicting...')
    inpainting_model.generator.eval()
    test_loader = dataset.create_iterator(batch_size=batch_size)

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    progbar = Progbar(total, width=20)
    for items in test_loader:
        images = items['image'].to(device)
        masks = items['mask'].to(device)
        constant_mask = items['constant_mask'].to(device)
        outputs, _, _ = inpainting_model.forward(images, masks, constant_mask)

        # Batch saving
        filename = items['filename']
        for f, result in zip(filename, outputs): 
            result = result[:, :config['dataset']['image_height'], :config['dataset']['image_width']]
            save_image(result, os.path.join(pred_path, f))

        progbar.add(len(images))
        
    mean_psnr, mean_l1, metrics = compute_metrics(pred_path, config['path']['test']['labels'])
    print('PSNR', mean_psnr)
    print('L1', mean_l1)

# ARGS
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path')
    parser.add_argument('--pred_path')
    parser.add_argument('--config_path')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--experiment_path', default='../experiments')
    args = parser.parse_args()
    main(args.images_path, args.pred_path, 
         args.config_path, args.experiment_path, 
         args.batch_size)