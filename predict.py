import os 
import yaml 
import random
import argparse

import cv2
import numpy as np
from PIL import Image


os.system("clear")
def main(pred_path, config_path, 
         images_path, masks_path,
         checkpoints_path, labels_path,
         blured,
         cuda, num_workers, batch_size):
    
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
    if cuda:
        config['gpu'] = cuda
    config['dataset']['num_workers'] = num_workers

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
    from torchvision.utils import save_image, make_grid
    from torchvision.transforms import ToPILImage

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

    # dataset 
    dataset = Dataset(config, training=False)
    test_loader = dataset.create_iterator(batch_size=batch_size)
    
    total = len(dataset)
    if total == 0:
        raise Exception("Dataset is empty!")
    
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
 
    # build the model and initialize
    generator = InpaintingGenerator(config).to(device)
    generator = nn.DataParallel(generator)

    checkpoints = os.listdir(checkpoints_path)
    if len(checkpoints) == 1:
        checkpoint = os.path.join(checkpoints_path, checkpoints[0])
        if config['gpu'] and torch.cuda.is_available():
            data = torch.load(checkpoint)
        else:
            data = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        
        generator.load_state_dict(data['generator'], strict=False)

    print('Predicting...')
    generator.eval()

    progbar = Progbar(total, width=50)
    for items in test_loader:
        images = items['image'].to(device)
        masks = items['mask'].to(device)
        constant_mask = items['constant_mask'].to(device)

        bs, c, h, w = images.size()
        outputs = np.zeros((bs, h, w, c))

        # predict
        if len(checkpoints) > 1:
            for ch in checkpoints:
                checkpoint = os.path.join(checkpoints_path, ch)
                if config['gpu'] and torch.cuda.is_available():
                    data = torch.load(checkpoint)
                else:
                    data = torch.load(checkpoint, map_location=lambda storage, loc: storage)
                
                generator.load_state_dict(data['generator'], strict=False)
                generator.eval()

                for i,result in enumerate(generator.module.predict(images, masks, constant_mask)):
                    grid = make_grid(result, nrow=8, padding=2, pad_value=0, normalize=False, range=None, scale_each=False)
                    result = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    outputs[i] += result    
        else:
            for i,result in enumerate(generator.module.predict(images, masks, constant_mask)):
                    grid = make_grid(result, nrow=8, padding=2, pad_value=0, normalize=False, range=None, scale_each=False)
                    result = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    outputs[i] += result

        outputs = outputs / len(checkpoints)
        outputs = np.array(outputs, dtype=np.uint8)

        # Batch saving
        filename = items['filename']
        for f, result in zip(filename, outputs): 
            result = result[:config['dataset']['image_height'], :config['dataset']['image_width']]
            
            if blured:
                test_img = np.array(Image.open(os.path.join(images_path, f)))

                mask_img = np.array(Image.open(os.path.join(masks_path, f)))
                mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)
                mask_img = (~np.array(mask_img, dtype=bool)) 
            
                test_img = test_img*mask_img
                for i in [3,5]:
                    result = cv2.blur(result,(i,i))

                result = result*(~mask_img)

                result = test_img + result
                result = Image.fromarray(result)
                result.save(os.path.join(pred_path, f))
            else:
                result = Image.fromarray(result)
                result.save(os.path.join(pred_path, f))

        progbar.add(len(images))
    
    if labels_path:
        compute_metrics(pred_path, labels_path)

# ARGS
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--checkpoints_path', required=True)

    parser.add_argument('--images_path', default=None)
    parser.add_argument('--masks_path', default=None)
    parser.add_argument('--labels_path', default=None)

    parser.add_argument('--blured', default=None)

    parser.add_argument('--cuda', default=None)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=1, type=int)

    args = parser.parse_args()
    main(args.pred_path, args.config_path, 
         args.images_path, args.masks_path,
         args.checkpoints_path, args.labels_path,
         args.blured,
         args.cuda, args.num_workers, args.batch_size)
