import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset_class import BasicDataset
from unet import UNet

from utils.utils import unnormalize_target
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path

def predict_img(net,
                dataloader,
                device, 
                target_norms, 
                predictions_dir):

    net.eval()
    num_val_batches = len(dataloader)

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Making predictions', unit='batch', leave=False):
        image, name = batch['image'], batch['name']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            pred = net(image)
            
            for i, output in enumerate(pred):
                logging.info(f'\nPredicting image {name} ...')

                # put predictions back into native format
                output = unnormalize_target(output, target_norms)

                # un-tensorify
                output = output.numpy()
                output = output.squeeze()
                output = Image.fromarray(output)

                # save prediction
                out_filename = os.path.join(predictions_dir, name[i] + ".tif")
                output.save(out_filename)

                logging.info(f'Predictions saved to {out_filename}')

    return 

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--config', help='Path to config file', default='configs/base.yaml')
    parser.add_argument('--model_epoch', '-m', default='last', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--split', default='val', help='The split to make predictions for (train, val, or test)')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    loader_args = dict(batch_size=1, num_workers=8, pin_memory=True)

    val_set = BasicDataset(cfg, args.split)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    net = UNet(n_channels=4, n_classes=1, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_epoch == 'last':
        model_epoch = cfg['epochs']
    else:
         model_epoch = args.model_epoch
    logging.info(f'Loading model checkpoints/checkpoint_epoch{model_epoch}.pth')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join(cfg['experiment_dir'], f'checkpoints/checkpoint_epoch{model_epoch}.pth'), map_location=device))

    logging.info('Model loaded!')

    # get the directory to save predictions to
    predictions_dir = os.path.join(cfg['experiment_dir'], 'predictions', str(args.split))
    os.makedirs(predictions_dir, exist_ok=True)

    # get the normalizations to put the outputs back in their native format
    target_norm_loc=Path(cfg['target_norm_loc'])
    target_norms = pd.read_csv(target_norm_loc, delim_whitespace=True).mean()

    predict_img(net, val_loader, device, target_norms, predictions_dir)
