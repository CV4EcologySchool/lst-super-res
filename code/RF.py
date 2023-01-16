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
    print('Target norms', target_norms)
    metrics_df = pd.DataFrame(columns=['file', 'landcover', 'old_r2_pred', 'old_rmse_pred','r2_pred', 'rmse_pred'])

    rmse = np.zeros(shape= len(val_loader))
    r2 = np.zeros(shape= len(val_loader))
    i = 0
    for batch in tqdm(val_loader, total=num_val_batches, desc='Making predictions', unit='batch', leave=False):
        image, name, target_input, landcover = batch['image'], batch['name'], batch['input_target'], batch['landcover']

        image = image.cpu().numpy()
        image = image.squeeze()
        print(name)
        coarse = image[0]
        length_c = coarse.shape[0] * coarse.shape[1]
        ground_truth = np.array(Image.open(os.path.join(output_target, ''.join(name) + ".tif")))
        old_ground_truth = ground_truth.flatten()
        old_ground_truth = old_ground_truth[~np.isnan(old_ground_truth)]
        ground_truth = normalize_target(old_ground_truth, target_norms, mean_for_nans=False)
        R = image[1]
        G = image[2]
        B = image[3]
        df = pd.DataFrame(data = {'coarse' : coarse.flatten(), 'R': R.flatten(), 'G': G.flatten(), 'B': B.flatten(), 'ground_truth': ground_truth})
        df = df.dropna()
        print(df)
        regressor = RandomForestRegressor()
        regressor.fit(df.iloc[:,1:4],df.iloc[:,0])
        y_pred = regressor.predict(df.iloc[:,1:4])
        

        print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(df['ground_truth'], y_pred)))
        rmse[i] = np.sqrt(metrics.mean_squared_error(df['ground_truth'], y_pred))
        print('R^2 score:', metrics.r2_score(df['ground_truth'], y_pred))
        r2[i] = metrics.r2_score(df['ground_truth'], y_pred)
        
        ground_truth = df['ground_truth']
        y_pred = (y_pred*target_norms['sd']) + target_norms['mean']
        ground_truth = (ground_truth*target_norms['sd']) + target_norms['mean']


        new_r2_pred = round(metrics.r2_score(ground_truth[~np.isnan(ground_truth)], y_pred[~np.isnan(y_pred)]), 2)
        new_rmse_pred = round(np.sqrt(metrics.mean_squared_error(ground_truth[~np.isnan(ground_truth)], y_pred[~np.isnan(y_pred)])), 2)
        print('New RMSE:', new_rmse_pred)
        print('New R^2:', new_r2_pred)

        df2 = pd.DataFrame({
        'file': [name], 
        'landcover': [landcover], 
        'old_r2_pred': [r2[i]], 
        'old_rmse_pred': [rmse[i]],  
        'r2_pred': [new_r2_pred],
        'rmse_pred': [new_rmse_pred]
        })
        metrics_df = metrics_df.append(df2, ignore_index = True)

    metrics_df.to_csv(os.path.join('RF','results_3.csv'))
    end = time.time()-start
    print('End Time:', end)