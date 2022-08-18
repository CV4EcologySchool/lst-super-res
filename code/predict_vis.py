# Visualize model outputs

# This script loads in either val or test data and creates predictions using the trained model of choice. 
# These predictions are plotted and evaluated using MSE and R2_score metrics. 

from dataset_class import BasicDataset
import argparse
import yaml
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

config = 'configs/base.yaml'

print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))
input_basemap = cfg["input_basemap"]
input_target = cfg["input_target"]
output_target = cfg["output_target"]

# the predictions of interest
predictions_dir = os.path.join(cfg['experiment_dir'], 'predictions')

#get most recent split with info about the landcover
splits_loc = cfg['splits_loc']

split_files = [file for file in os.listdir(splits_loc) if file.endswith(".csv")] # list all the different splits
recent_split = sorted(split_files, key=lambda fn:os.path.getctime(os.path.join(splits_loc, fn)))[-1] # get most recent split
split_file = pd.read_csv(os.path.join(splits_loc, recent_split))

def get_r2(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    return round(r2_score(img_1[~np.isnan(img_1)], img2[~np.isnan(img2)]), 2)

def get_mse(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    return round(mean_squared_error(img_1[~np.isnan(img_1)], img2[~np.isnan(img2)]), 2)

def get_mae(img_1, img2): # this assumes values to be ignored are already masked out (are np.nan) and the images are a numpy array
    return round(mean_absolute_error(img_1[~np.isnan(img_1)], img2[~np.isnan(img2)]), 2)

# dataframe of evaluation metrics
metrics_df = pd.DataFrame(columns=['file', 'landcover', 'r2_pred', 'mse_pred', 'mae_pred', 'r2_coarse', 'mse_coarse', 'mae_coarse'])

for image in os.listdir(predictions_dir):
    # get the landcover
    landcover = split_file[split_file['tiles'] == image]["Main Cover"]._values[0]

    plt.figure(figsize=(24,6))

    ground_truth = np.array(Image.open(os.path.join(output_target, image)))
    mask = ground_truth < 0
    ground_truth[mask] = np.nan
    plt.subplot(1, 4, 1)
    plt.imshow(ground_truth, vmin = 268, vmax=353)
    plt.title(str(image))
    plt.axis("off")


    pred = np.array(Image.open(os.path.join(predictions_dir, image)))
    pred[mask] = np.nan
    r2_pred = get_r2(ground_truth, pred)
    mse_pred = get_mse(ground_truth, pred)
    mae_pred = get_mae(ground_truth, pred)
    plt.subplot(1, 4, 2)
    plt.imshow(pred, vmin = 268, vmax=353)
    plt.title(f'Prediction: r2 = {str(r2_pred)}, mse = {str(mse_pred)}, mae = {str(mae_pred)}')
    plt.axis("off")


    coarse = np.array(Image.open(os.path.join(input_target, image)))
    coarse[mask] = np.nan
    r2_coarse = get_r2(ground_truth, coarse)
    mse_coarse = get_mse(ground_truth, coarse)
    mae_coarse = get_mae(ground_truth, coarse)
    plt.subplot(1, 4, 3)
    plt.imshow(coarse, vmin = 268, vmax=353)
    plt.title(f'Coarsened input: r2 = {str(r2_coarse)}, mse = {str(mse_coarse)}, mae = {str(mae_coarse)}')
    plt.axis("off")


    rgb = np.array(Image.open(os.path.join(input_basemap, image)))
    plt.subplot(1, 4, 4)
    plt.imshow(rgb)
    plt.title(str(landcover))
    plt.axis("off")

    os.makedirs(os.path.join(cfg['experiment_dir'], "prediction_plots"), exist_ok=True)
    plt.savefig(os.path.join(cfg['experiment_dir'], "prediction_plots", str(image).split(".tif")[0]+".png"))
    # plt.show() # for the notebook version

    # add the evaluation metrics to a pandas dataframe
    df2 = pd.DataFrame({
        'file': [image], 
        'landcover': [landcover], 
        'r2_pred': [r2_pred], 
        'mse_pred': [mse_pred], 
        'mae_pred': [mae_pred], 
        'r2_coarse': [r2_coarse], 
        'mse_coarse': [mse_coarse], 
        'mae_coarse': [mae_coarse]
        })
    metrics_df = metrics_df.append(df2, ignore_index = True)


# save the pandas dataframe of evaluation metrics as csv
metrics_df.to_csv(os.path.join(cfg['experiment_dir'], "prediction_metrics.csv"))

# Print out the average metrics
print(metrics_df.mean())