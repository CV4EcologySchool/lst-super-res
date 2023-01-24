'''
    Visualize model outputs:
    This script loads in either val or test data and creates predictions using the trained model of choice. 
    These predictions are plotted and evaluated using MSE and R2_score metrics. 

    2022 Anna Boser
'''
from dataset_class import BasicDataset
import argparse
import yaml
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--config', help='Path to config file', default='configs/base.yaml')
    parser.add_argument('--split', default='val', help='The split to make predictions for (train, val, or test)')

    return parser.parse_args()
args = get_args()

config = args.config

print(f'Using config "{config}"')
cfg = yaml.safe_load(open(config, 'r'))
input_basemap = cfg["input_basemap"]
input_target = cfg["input_target"]
output_target = cfg["output_target"]

# the predictions of interest
predictions_dir = os.path.join(cfg['experiment_dir'], 'predictions', str(args.split))

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
metrics_df = pd.DataFrame(columns=['file', 'landcover', 'r2_pred', 'mse_pred', 'mae_pred', 'r2_coarse', 'mse_coarse', 'mae_coarse', 'ssim', 'psnr'])

for image in os.listdir(predictions_dir):
    # get the landcover
    landcover = split_file[split_file['tiles'] == image]["Main Cover"]._values[0]

    plt.figure(figsize=(24,6))

    # visualize ground truth image
    ground_truth = np.array(Image.open(os.path.join(output_target, image)))
    mask = ground_truth < 0
    ground_truth[mask] = np.nan
    print('Ground Truth min:', np.nanmin(ground_truth), flush = True)
    print('Ground Truth max:', np.nanmax(ground_truth), flush = True)
    plt.subplot(1, 4, 1)
    plt.imshow(ground_truth, vmin = np.nanmin(ground_truth), vmax= np.nanmax(ground_truth), cmap = 'coolwarm') 
    plt.title(str(image))
    plt.axis("off")

    # visualize prediction image
    pred = np.array(Image.open(os.path.join(predictions_dir, image)))
    pred[mask] = np.nan
    r2_pred = get_r2(ground_truth, pred)
    mse_pred = get_mse(ground_truth, pred)
    ssim = ssim(ground_truth, pred)
    psnr = psnr(ground_truth, pred)
    print('Name:', image, flush = True)
    print('R2 pred:', r2_pred, flush = True)
    print('MSE pred:', mse_pred, flush = True)
    print('SSIM:', ssim, flush = True)
    print('PSNR:', psnr, flush = True)
    mae_pred = get_mae(ground_truth, pred)
    plt.subplot(1, 4, 2)
    plt.imshow(pred, vmin = np.nanmin(ground_truth), vmax= np.nanmax(ground_truth), cmap = 'coolwarm') 
    plt.title(f'Prediction: r2 = {str(r2_pred)}, mse = {str(mse_pred)}, mae = {str(mae_pred)}')
    plt.axis("off")

    # visualize coarse image
    coarse = np.array(Image.open(os.path.join(input_target, image)))
    coarse[mask] = np.nan
    r2_coarse = get_r2(ground_truth, coarse)
    mse_coarse = get_mse(ground_truth, coarse)
    mae_coarse = get_mae(ground_truth, coarse)
    plt.subplot(1, 4, 3)
    plt.imshow(coarse, vmin = np.nanmin(ground_truth), vmax= np.nanmax(ground_truth), cmap = 'coolwarm') 
    plt.title(f'Coarsened input: r2 = {str(r2_coarse)}, mse = {str(mse_coarse)}, mae = {str(mae_coarse)}')
    plt.axis("off")

    # visualize RGB image
    rgb = np.array(Image.open(os.path.join(input_basemap, image)))
    plt.subplot(1, 4, 4)
    plt.imshow(rgb)
    plt.title(str(landcover))
    plt.axis("off")

    # create a directory to store prediction plots
    os.makedirs(os.path.join(cfg['experiment_dir'], "prediction_plots", str(args.split)), exist_ok=True)
    # plt.savefig(os.path.join(cfg['experiment_dir'], "prediction_plots", str(args.split), str(image).split(".tif")[0]+".png"))
    # plt.show() # for the notebook version
    plt.close('all')

    # add the evaluation metrics to a pandas dataframe
    df2 = pd.DataFrame({
        'file': [image], 
        'landcover': [landcover], 
        'r2_pred': [r2_pred], 
        'mse_pred': [mse_pred], 
        'mae_pred': [mae_pred], 
        'r2_coarse': [r2_coarse], 
        'mse_coarse': [mse_coarse], 
        'mae_coarse': [mae_coarse],
        'ssim': [ssim],
        'psnr': [psnr]
        })
    metrics_df = metrics_df.append(df2, ignore_index = True)


# save the pandas dataframe of evaluation metrics as csv
os.makedirs(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(args.split)))
metrics_df.to_csv(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(args.split), "prediction_metrics.csv"))

# Print out the average metrics
metrics_df.mean().to_csv(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(args.split), "prediction_metrics_mean.csv"))