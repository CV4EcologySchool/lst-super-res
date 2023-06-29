'''
    Visualize model outputs:
    This script loads in either val or test data and creates predictions using the trained model of choice. 
    These predictions are plotted and evaluated using MSE, SSIM, and R2_score metrics. 

    2022 Anna Boser
'''




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

    # visualize ground truth image
    ground_truth_im = Image.open(os.path.join(output_target, image))
    ground_truth = np.array(ground_truth_im)
    mask = ground_truth < 0
    ground_truth[mask] = np.nan
    print('Ground Truth min:', np.nanmin(ground_truth), flush = True)
    print('Ground Truth max:', np.nanmax(ground_truth), flush = True)
    plt.subplot(1, 4, 1)
    plt.imshow(ground_truth, vmin = np.nanmin(ground_truth), vmax= np.nanmax(ground_truth), cmap = 'coolwarm') 
    plt.title(str(image))
    plt.axis("off")

    # visualize prediction image
    pred_im = Image.open(os.path.join(predictions_dir, image))
    pred = np.array(pred_im)
    pred[mask] = np.nan
    r2_pred = get_r2(ground_truth, pred)
    mse_pred = get_mse(ground_truth, pred)
    print(pred.shape, ground_truth.shape, flush = True)
    ssim_pred = get_ssim(ground_truth, pred)
    print('Name:', image, flush = True)
    print('R2 pred:', r2_pred, flush = True)
    print('MSE pred:', mse_pred, flush = True)
    print('SSIM pred:', ssim_pred, flush = True)
    mae_pred = get_mae(ground_truth, pred)
    plt.subplot(1, 4, 2)
    plt.imshow(pred, vmin = np.nanmin(ground_truth), vmax= np.nanmax(ground_truth), cmap = 'coolwarm') 
    plt.title(f'Prediction: r2 = {str(r2_pred)}, mse = {str(mse_pred)}, mae = {str(mae_pred)}')
    plt.axis("off")

    # visualize coarse image
    coarse = np.array(Image.open(os.path.join(input_target, image)))
    coarse[mask] = np.nan
    ssim_coarse = get_ssim(ground_truth, coarse)
    print('SSIM coarse:', ssim_coarse, flush = True)
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
    plt.savefig(os.path.join(cfg['experiment_dir'], "prediction_plots", str(args.split), str(image).split(".tif")[0]+".png"))
    # plt.show() # for the notebook version
    plt.close('all')

    # add the evaluation metrics to a pandas dataframe
    df2 = pd.DataFrame({
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
    print('done one!', flush = True)
    metrics_df = metrics_df.append(df2, ignore_index = True)


# save the pandas dataframe of evaluation metrics as csv
os.makedirs(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(args.split)))
metrics_df.to_csv(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(args.split), "prediction_metrics.csv"))

# Print out the average metrics
metrics_df.mean().to_csv(os.path.join(cfg['experiment_dir'], 'prediction_metrics', str(args.split), "prediction_metrics_mean.csv"))