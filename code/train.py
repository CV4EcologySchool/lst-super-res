import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_class import BasicDataset
from evaluate import evaluate
from unet import UNet

import pandas as pd
import shutil

from utils.utils import unnormalize_basemap

import yaml
import os

def train_net(cfg,
              net,
              device,
              save_checkpoint: bool = True,
              amp: bool = False):

    # where to save the trained model and configurations
    dir_checkpoint = Path(os.path.join(cfg['experiment_dir'], 'checkpoints'))

    # get hyperparameters from configurations
    epochs = cfg['epochs']
    epochs_done = cfg['epochs_done']
    all_epochs = epochs + epochs_done
    batch_size = cfg['batch_size']
    learning_rate = float(cfg['learning_rate'])
    
    # 1. Create dataset
    train_set = BasicDataset(cfg, 'train')
    val_set = BasicDataset(cfg, 'val')

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  save_checkpoint=save_checkpoint, 
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize MSE
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss(reduction='none')
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        epoch += epochs_done
        net.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{all_epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_labels = batch['label']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                assert torch.logical_not(torch.isnan(images).any()), \
                    f'NaN value detected in input image.' \

                images = images.to(device=device, dtype=torch.float32)
                true_labels = true_labels.to(device=device, dtype=torch.float32)

                mask = torch.isnan(true_labels)

                with torch.cuda.amp.autocast(enabled=amp):
                    pred = net(images)
                    true_labels[mask] = -3.4e+30 # set missing values to any real number -- doesn't matter because it will be ignored in the loss. 
                    loss = criterion(pred, true_labels)
                    loss[mask] = 0             # ignore pixel locations where target LST is NaN
                    loss = loss.mean()

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            
        # Evaluation round
        # division_step = (len(train_loader) // (10 * batch_size))
        # if division_step > 0:
        #     if global_step % division_step == 0:
        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        val_score = evaluate(net, val_loader, device)
        scheduler.step(val_score)

        # true labels for plotting -- put zero where there's a nan
        mask = torch.isnan(true_labels)
        true_labels_plot = true_labels.clone()
        true_labels_plot[mask] = 0

        logging.info('Validation score: {}'.format(val_score))
        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation': val_score,
            'images': {
                'basemap': wandb.Image(images[0,1:,:,:].cpu(), mode="RGB"),
                'lst': wandb.Image(images[0,0,:,:].float().cpu())
            },
            'labels': {
                'true': wandb.Image(true_labels_plot[0].float().cpu()),
                'pred': wandb.Image(pred[0].float().cpu())
            },
            'step': global_step,
            'epoch': epoch,
            **histograms
        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            #if epoch > 1:
            #    old_checkpoint = os.path.join(cfg['experiment_dir'], 'checkpoints/')
            #    shutil.rmtree(old_checkpoint)
            #    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target labels')
    parser.add_argument('--config', help='Path to config file', default='configs/base.yaml')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')



    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=4, n_classes=1, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    # save the configurations for this experiment in the experiment folder
    experiments_dir = cfg['experiment_dir']
    os.makedirs(experiments_dir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(experiments_dir, "configs.yaml"))

    # also save the split and split info to the experiments folder
    splits_loc = cfg["splits_loc"]
    split_files = [file for file in os.listdir(splits_loc) if file.endswith(".csv")]
    recent_split = sorted(split_files, key=lambda fn:os.path.getctime(os.path.join(splits_loc, fn)))[-1] # get most recent split
    shutil.copyfile(os.path.join(splits_loc, recent_split), os.path.join(experiments_dir, "split.csv")) # copy the split
    shutil.copyfile(os.path.join(splits_loc, "info", recent_split.split(".csv")[0] + ".txt"), os.path.join(experiments_dir, "split_info.csv")) # copy the split info

    net.to(device=device)
    try:
        train_net(cfg=cfg,
                  net=net,
                  device=device,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
