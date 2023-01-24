'''
    This script enhances coarsened LST images using a Random Forest regressor.

    2022 Ryan Stofer
'''
import argparse
import logging
import os
import time
from tkinter import image_types

import numpy as np
from PIL import Image

from dataset_class import BasicDataset

import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from utils.utils import normalize_target

# Add get arguments function
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--config', help='Path to config file', default='configs/base.yaml')
    parser.add_argument('--model_epoch', '-m', default='last', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--split', default='val', help='The split to make predictions for (train, val, or test)')

    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()

    args = get_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    loader_args = dict(batch_size=1, num_workers=8, pin_memory=True)

    print(f'Using Args: "{args.split}"')
    val_set = BasicDataset(cfg, args.split, predict="True")
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    num_val_batches = len(val_loader)

    # get the directory to save predictions to
    predictions_dir = os.path.join('RF', 'predictions', str(args.split))
    os.makedirs(predictions_dir, exist_ok=True)

    output_target = cfg["output_target"]
    target_norms = pd.read_csv(Path(cfg['target_norm_loc']), delim_whitespace=True).mean()
    metrics_df = pd.DataFrame(columns=['file', 'landcover', 'r2_pred', 'rmse_pred'])

    rmse = np.zeros(shape= len(val_loader))
    r2 = np.zeros(shape= len(val_loader))
    i = 0
    for batch in tqdm(val_loader, total=num_val_batches, desc='Making predictions', unit='batch', leave=False):
        # Get each individual image data and metadata
        image, name, target_input, landcover = batch['image'], batch['name'], batch['input_target'], batch['landcover']

        # convert for CPU usage
        image = image.cpu().numpy()
        image = image.squeeze()

        # Initialize data frame with coarsened image along with RGB and ground truth values
        coarse = image[0]
        ground_truth = np.array(Image.open(os.path.join(output_target, ''.join(name) + ".tif"))).flatten()
        ground_truth = ground_truth[~np.isnan(ground_truth)]
        ground_truth = normalize_target(ground_truth, target_norms, mean_for_nans=False)
        R = image[1]
        G = image[2]
        B = image[3]
        df = pd.DataFrame(data = {'coarse' : coarse.flatten(), 'R': R.flatten(), 'G': G.flatten(), 'B': B.flatten(), 'ground_truth': ground_truth})
        df = df.dropna()

        # Fits regressor to image
        regressor = RandomForestRegressor()
        regressor.fit(df.iloc[:,1:4],df.iloc[:,0])
        y_pred = regressor.predict(df.iloc[:,1:4])
        
        # Compute relevant metrics after unnormalizing prediction and ground truth
        ground_truth = df['ground_truth']
        y_pred = (y_pred*target_norms['sd']) + target_norms['mean']
        ground_truth = (ground_truth*target_norms['sd']) + target_norms['mean']

        r2_pred = round(metrics.r2_score(ground_truth[~np.isnan(ground_truth)], y_pred[~np.isnan(y_pred)]), 2)
        rmse_pred = round(np.sqrt(metrics.mean_squared_error(ground_truth[~np.isnan(ground_truth)], y_pred[~np.isnan(y_pred)])), 2)

        # Generate dataframe to save results
        df2 = pd.DataFrame({
        'file': [name], 
        'landcover': [landcover],   
        'r2_pred': [r2_pred],
        'rmse_pred': [rmse_pred]
        })
        metrics_df = metrics_df.append(df2, ignore_index = True)

    # Save dataframe as .csv file
    metrics_df.to_csv(os.path.join('RF','results.csv'))
    end = time.time()-start
    print('End Time:', end)