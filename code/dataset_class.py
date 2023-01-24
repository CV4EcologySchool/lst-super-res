'''
    This script creates the dataset class to read and process data to feed into the dataloader. 
    The Dataset class is called for both model training and predicting in train.py and predict.py respectively.

    2022 Anna Boser
'''

import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A

from utils.utils import normalize_target
from utils.utils import normalize_basemap

# dataset class

class BasicDataset(Dataset):
    def __init__(self, cfg, split, predict="False"):

        self.split = split

        self.predict = predict

        self.transform = A.Compose([
            A.HorizontalFlip(p=cfg['HorizontalFlip']),
            A.VerticalFlip(p=cfg['VerticalFlip']),
            A.RandomRotate90(p=cfg['RandomRotate90'])
        ])

        self.toTensor = ToTensor()

        # data paths
        self.input_basemap = Path(cfg['input_basemap'])
        self.input_target = Path(cfg['input_target'])
        self.output_target = Path(cfg['output_target'])
        self.target_norm_loc = Path(cfg['target_norm_loc'])
        self.basemap_norm_loc = Path(cfg['basemap_norm_loc'])
        self.residual = cfg['Residual']

        # check that images are in the given directories
        if not [file for file in listdir(self.input_basemap) if not file.startswith('.')]:
            raise RuntimeError(f'No input file found in {self.input_basemap}, make sure you put your images there')
        if not [file for file in listdir(self.input_target) if not file.startswith('.')]:
            raise RuntimeError(f'No input file found in {self.input_target}, make sure you put your images there')
        if not [file for file in listdir(self.output_target) if not file.startswith('.')]:
            raise RuntimeError(f'No input file found in {self.output_target}, make sure you put your images there')

        # normalization factors for preprocessing
        self.target_norms = pd.read_csv(self.target_norm_loc, delim_whitespace=True).mean()

        self.basemap_norms = pd.read_csv(self.basemap_norm_loc, delim_whitespace=True).mean()

        # get the ids in split
        self.splits_loc = cfg['splits_loc']

        split_files = [file for file in os.listdir(self.splits_loc) if file.endswith(".csv")]
        recent_split = sorted(split_files, key=lambda fn:os.path.getctime(os.path.join(self.splits_loc, fn)))[-1] # get most recent split
        split_file = pd.read_csv(os.path.join(self.splits_loc, recent_split))

        self.ids = [splitext(file)[0] for file in split_file[split_file['split'] == self.split]['tiles']]

        self.split_file = split_file # use this later when you get the main landcover of a tile

        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)

    def preprocess(self, input_basemap_im, input_target_im, output_target_im):

        # turn basemap and target into tensors
        input_basemap_im = self.toTensor(input_basemap_im)*255 # the default is 0 to 255 turned into 0 to 1 -- override this since I'm doing my own normalizations
        input_target_im = self.toTensor(input_target_im) # no conversion. NA is -3.4e+38
        output_target_im = self.toTensor(output_target_im) # no conversion. NA is -3.4e+38

        # if statement here based on if we want to train on the residual of the images
        if self.residual:
            output = normalize_target(output_target_im, self.target_norms, mean_for_nans=False) - normalize_target(input_target_im, self.target_norms, mean_for_nans=True)
        else:
            output = normalize_target(output_target_im, self.target_norms, mean_for_nans=False)

        input_target = normalize_target(input_target_im, self.target_norms, mean_for_nans=True)
        ib1, ib2, ib3 = normalize_basemap(input_basemap_im, self.basemap_norms, n_bands=3)

        input = torch.cat([input_target, ib1, ib2, ib3], dim=0)

        return input_target, input, output

    @staticmethod
    def load(filename, bands):
        if bands == 1:
            return Image.open(filename)
        elif bands == 3:
            return Image.open(filename).convert('RGB') # HELP: This works fine for an 8 bit image, but somehow the tifs I have were saved as floats even though they definitely could just be 8-bit (that was thier original format before cropping)
        else: 
            raise Exception('Image must be one or three bands')

    def __getitem__(self, idx):
        name = self.ids[idx]
        input_basemap_im = list(self.input_basemap.glob(name + '.tif*'))
        input_target_im = list(self.input_target.glob(name + '.tif*'))
        output_target_im = list(self.output_target.glob(name + '.tif*'))

        assert len(input_basemap_im) == 1, f'Either no basemap input or multiple basemap inputs found for the ID {name}: {input_basemap_im}'
        assert len(input_target_im) == 1, f'Either no target input or multiple target inputs found for the ID {name}: {input_target_im}'
        assert len(output_target_im) == 1, f'Either no target output or multiple target outputs found for the ID {name}: {output_target_im}'
        input_basemap_im = self.load(input_basemap_im[0], bands = 3)
        input_target_im = self.load(input_target_im[0], bands = 1)
        output_target_im = self.load(output_target_im[0], bands = 1)

        assert input_basemap_im.size == input_target_im.size, \
            f'Target and basemap input {name} should be the same size, but are {input_target_im.size} and {input_basemap_im.size}'
        assert input_target_im.size == output_target_im.size, \
            f'Input and output {name} should be the same size, but are {input_target_im.size} and {output_target_im.size}'

        input_target, input, output = self.preprocess(input_basemap_im, input_target_im, output_target_im)
        
        if self.split == "train":
            if self.predict == "False":
                transforms = self.transform(image=input.numpy().transpose(1,2,0), mask=output.numpy().transpose(1,2,0))
                input = transforms['image']
                output = transforms['mask']
                input, output = torch.from_numpy(input.transpose(2,0,1)), torch.from_numpy(output.transpose(2,0,1)) 

        # also get the main class of this particular image
        landcover = self.split_file[self.split_file['tiles'] == name + ".tif"]["Main Cover"]._values[0]

        return {
            'image': input,
            'input_target': input_target,
            'label': output, 
            'name': name,
            'landcover': landcover
        }


if __name__ == '__main__':
    # test code
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/base.yaml')
    args = parser.parse_args()
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    dataset = BasicDataset(cfg, 'train')
    len(dataset)
    [dataset.__getitem__(i) for i in range(len(dataset))]
    dataset = BasicDataset(cfg, 'val')
    len(dataset)
    [dataset.__getitem__(i) for i in range(len(dataset))]
    dataset = BasicDataset(cfg, 'test')
    len(dataset)
    [dataset.__getitem__(i) for i in range(len(dataset))]