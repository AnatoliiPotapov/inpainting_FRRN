import os 
import yaml 
import random
import argparse

import numpy as np


os.system("clear")
def main(pred_path, config_path, 
         images_path, masks_path,
         checkpoint, labels_path):
    
    from model.net import InpaintingGenerator
    from utils.general import get_config
    from utils.progbar import Progbar
    from data.dataset import Dataset
    from scripts.metrics import compute_metrics

    # load config
    code_path = './'
    config, pretty_config = get_config(os.path.join(code_path, config_path))

    if images_path:
        config['path']['test']['images'] = images_path
    if masks_path:
        config['path']['test']['masks'] = masks_path

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

    # build the model and initialize
    generator = InpaintingGenerator(config).to(device)
    generator = nn.DataParallel(generator)
    if config['gpu'] and torch.cuda.is_available():
        data = torch.load(checkpoint)
    else:
        data = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    
    generator.load_state_dict(data['generator'], strict=False)

    # dataset 
    dataset = Dataset(config, training=False)

    # Train the generator
    total = len(dataset)
    if total == 0:
        raise Exception("Dataset is empty!")

    print('Predicting...')
    generator.eval()
    test_loader = dataset.create_iterator(batch_size=config['training']['batch_size'])

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    progbar = Progbar(total, width=50)
    for items in test_loader:
        images = items['image'].to(device)
        masks = items['mask'].to(device)
        constant_mask = items['constant_mask'].to(device)
        outputs = generator.module.predict(images, masks, constant_mask)

        # Batch saving
        filename = items['filename']
        for f, result in zip(filename, outputs): 
            result = result[:, :config['dataset']['image_height'], :config['dataset']['image_width']]
            save_image(result, os.path.join(pred_path, f))

        progbar.add(len(images))
    
    if labels_path:
        compute_metrics(pred_path, labels_path)

# ARGS
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--checkpoint', required=True)

    parser.add_argument('--images_path', default=None)
    parser.add_argument('--masks_path', default=None)
    parser.add_argument('--labels_path', default=None)

    args = parser.parse_args()
    main(args.pred_path, args.config_path, 
         args.images_path, args.masks_path,
         args.checkpoint, args.labels_path)