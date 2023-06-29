'''
    This script performs predictions using the trained UNet model on the validation set.

    2022 Anna Boser
'''
import argparse
import logging
import os
import time
from tkinter import image_types
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset_class import BasicDataset
from unet import UNet

from utils.utils import unnormalize_target
from utils.utils import load
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skimage import img_as_float
from skimage.metrics import structural_similarity

def predict_img(net,
                dataloader,
                device, 
                basemap_norms, 
                predictions_dir,
                residual):

    net.eval()
    num_val_batches = len(dataloader)
    print('num_val_batches:', num_val_batches)

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Making predictions', unit='batch', leave=False):
        image, name, target_input = batch['image'], batch['name'], batch['input_target']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        target_input = target_input.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            pred = net(image)
            
            for i, output in enumerate(pred):
                logging.info(f'\nPredicting image {name} ...')

                # put predictions back into native format
                # if we are calculating original values keep unnormalize_target
                # else if we are calculating residuals, add to coarse array
                if residual:
                    output = unnormalize_target(target_input+output, batch['target_mean'], batch['target_sd']) # image is a float 32 torch
                else:
                    output = unnormalize_target(output, batch['target_mean'], batch['target_sd'])

                # un-tensorify
                output = output.cpu().numpy()
                output = output.squeeze()
                output = Image.fromarray(output)

                # save prediction
                out_filename = os.path.join(predictions_dir, name[i] + ".tif")
                output.save(out_filename)

                logging.info(f'Predictions saved to {out_filename}')

                # return input and output target
                input_target = unnormalize_target(batch['input_target'], batch['target_mean'], batch['target_sd'])
                input_target = input_target.cpu().numpy()
                input_target = input_target.squeeze()
                input_target = Image.fromarray(input_target)

                label = unnormalize_target(batch['output_target'], batch['target_mean'], batch['target_sd'])
                label = label.cpu().numpy()
                label = label.squeeze()
                label = Image.fromarray(label)

                return input_target, label


def get_r2(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    return round(r2_score(img_1[~np.isnan(img_1)], img2[~np.isnan(img2)]), 2)

def get_mse(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    return round(mean_squared_error(img_1[~np.isnan(img_1)], img2[~np.isnan(img2)]), 2)

def get_mae(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    return round(mean_absolute_error(img_1[~np.isnan(img_1)], img2[~np.isnan(img2)]), 2)

def get_ssim(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    img_1 = img_1[~np.isnan(img_1)]
    img2 = img2[~np.isnan(img2)]
    result = structural_similarity(img_1, img2, data_range = img_1.max()-img_1.min()) 
    return round(result, 2)

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
    
    print(f'Using Args: "{args.split}"')

    random.seed(cfg['Seed'])

    val_set = BasicDataset(cfg, args.split, predict="True")
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    net = UNet(n_channels=4, n_classes=1, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
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

    pretrain = cfg['pretrain']
    if pretrain:
        basemap_norm_loc = Path(cfg['pretrain_basemap_norm_loc'])
        input_basemap = cfg["pretrain_basemap"]
        splits_loc = cfg["pretrain_splits_loc"]
    else:
        basemap_norm_loc = Path(cfg['basemap_norm_loc'])
        input_basemap = cfg["input_basemap"]
        splits_loc = cfg['splits_loc']
    basemap_norms = pd.read_csv(basemap_norm_loc, delim_whitespace=True).mean()
    # See if we calculate the residual instead
    residual = cfg['Residual']
    start = time.time()
    input_target, output_target = predict_img(net, val_loader, device, basemap_norms, predictions_dir, residual)
    print('input_target min:', np.nanmin(input_target), flush = True)
    print('input_target max:', np.nanmax(input_target), flush = True)
    print('output_target min:', np.nanmin(output_target), flush = True)
    print('output_target max:', np.nanmax(output_target), flush = True)
    end = time.time()-start
    print('Total prediction time:', end)

    # Should visualization plots be made
    visualize = cfg['visualize']

    split_files = [file for file in os.listdir(splits_loc) if file.endswith(".csv")] # list all the different splits
    recent_split = sorted(split_files, key=lambda fn:os.path.getctime(os.path.join(splits_loc, fn)))[-1] # get most recent split
    split_file = pd.read_csv(os.path.join(splits_loc, recent_split))


    # dataframe of evaluation metrics
    metrics_df = pd.DataFrame(columns=['file', 'landcover', 'r2_pred', 'mse_pred', 'mae_pred','ssim_pred','r2_coarse', 'mse_coarse', 'mae_coarse', 'ssim_coarse'])

    for image in os.listdir(predictions_dir):
        print('start', image, flush=True)
        # get the landcover
        landcover = split_file[split_file['tiles'] == image]["Main Cover"]._values[0]

        plt.figure(figsize=(24,6))

        # load in ground truth image
        ground_truth = np.array(output_target)
        if not pretrain:
            mask = ground_truth < 0
            ground_truth[mask] = np.nan
        print('Ground Truth min:', np.nanmin(ground_truth), flush = True)
        print('Ground Truth max:', np.nanmax(ground_truth), flush = True)

        # metrics of prediction image
        pred_im = Image.open(os.path.join(predictions_dir, image))
        pred = np.array(pred_im)
        if not pretrain:
            pred[mask] = np.nan
        r2_pred = get_r2(ground_truth, pred)
        mse_pred = get_mse(ground_truth, pred)
        ssim_pred = get_ssim(ground_truth, pred)
        mae_pred = get_mae(ground_truth, pred)


        # metrics of coarse image
        coarse = np.array(input_target)
        if not pretrain:
            coarse[mask] = np.nan
        print('Coarse min:', np.nanmin(coarse), flush = True)
        print('Coarse Truth max:', np.nanmax(coarse), flush = True)
        ssim_coarse = get_ssim(ground_truth, coarse)
        r2_coarse = get_r2(ground_truth, coarse)
        mse_coarse = get_mse(ground_truth, coarse)
        mae_coarse = get_mae(ground_truth, coarse)

        # visualize RGB image
        rgb = np.array(load(os.path.join(input_basemap, image), bands = 3))

        if visualize:
            plt.subplot(1, 4, 1)
            plt.imshow(ground_truth, vmin = np.nanmin(ground_truth), vmax= np.nanmax(ground_truth), cmap = 'coolwarm') 
            plt.title(str(image))
            plt.axis("off")
            plt.subplot(1, 4, 2)
            plt.imshow(pred, vmin = np.nanmin(ground_truth), vmax= np.nanmax(ground_truth), cmap = 'coolwarm') 
            plt.title(f'Prediction: r2 = {str(r2_pred)}, mse = {str(mse_pred)}, mae = {str(mae_pred)}')
            plt.axis("off")
            plt.subplot(1, 4, 3)
            plt.imshow(coarse, vmin = np.nanmin(coarse), vmax= np.nanmax(coarse), cmap = 'coolwarm') 
            plt.title(f'Coarsened input: r2 = {str(r2_coarse)}, mse = {str(mse_coarse)}, mae = {str(mae_coarse)}')
            plt.axis("off")
            plt.subplot(1, 4, 4)
            plt.imshow(rgb)
            plt.title(str(landcover))
            plt.axis("off")

            # create a directory to store prediction plots
            os.makedirs(os.path.join(cfg['experiment_dir'], "prediction_plots", str(args.split)), exist_ok=True)
            plt.savefig(os.path.join(cfg['experiment_dir'], "prediction_plots", str(args.split), str(image).split(".tif")[0]+".png"))
            # plt.show() # for the notebook version
            plt.close('all')

        # add the evaluation metrics to a pandas dataframe
        df = pd.DataFrame({
            'file': [image], 
            'landcover': [landcover], 
            'r2_pred': [r2_pred], 
            'mse_pred': [mse_pred], 
            'mae_pred': [mae_pred],
            'ssim_pred': [ssim_pred], 
            'r2_coarse': [r2_coarse], 
            'mse_coarse': [mse_coarse], 
            'mae_coarse': [mae_coarse],
            'ssim_coarse': [ssim_coarse]
            })
        print('Done:', df, flush = True)
        metrics_df = metrics_df.append(df, ignore_index = True)


    # save the pandas dataframe of evaluation metrics as csv
    os.makedirs(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(args.split)), exist_ok=True)
    metrics_df.to_csv(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(args.split), "prediction_metrics.csv"))

    # Save the average metrics as well
    metrics_df.mean().to_csv(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(args.split), "prediction_metrics_mean.csv"))