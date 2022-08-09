'''
    This script creates the dataset class to read and process data to feed into the dataloader. 

    2022 Anna Boser
'''

import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

# dataset class

class BasicDataset(Dataset):
    def __init__(self, cfg, split):

        self.split = split

        self.toTensor = ToTensor()

        # data paths
        self.input_basemap = Path(cfg['input_basemap'])
        self.input_target = Path(cfg['input_target'])
        self.output_target = Path(cfg['output_target'])
        self.target_norm_loc = Path(cfg['target_norm_loc'])
        self.basemap_norm_loc = Path(cfg['basemap_norm_loc'])

        # check that images are in the given directories
        if not [file for file in listdir(self.input_basemap) if not file.startswith('.')]:
            raise RuntimeError(f'No input file found in {self.input_basemap}, make sure you put your images there')
        if not [file for file in listdir(self.input_target) if not file.startswith('.')]:
            raise RuntimeError(f'No input file found in {self.input_target}, make sure you put your images there')
        if not [file for file in listdir(self.output_target) if not file.startswith('.')]:
            raise RuntimeError(f'No input file found in {self.output_target}, make sure you put your images there')

        # normalization factors for preprocessing
        target_norms = pd.read_csv(self.target_norm_loc, delim_whitespace=True).mean()
        self.target_mean = target_norms['mean']
        self.target_sd = target_norms['sd']

        basemap_norms = pd.read_csv(self.basemap_norm_loc, delim_whitespace=True).mean()
        self.basemap_mean1 = basemap_norms['mean1']
        self.basemap_mean2 = basemap_norms['mean2']
        self.basemap_mean3 = basemap_norms['mean3']
        self.basemap_sd1 = basemap_norms['sd1']
        self.basemap_sd2 = basemap_norms['sd2']
        self.basemap_sd3 = basemap_norms['sd3']

        # get the ids in split
        self.splits_loc = cfg['splits_loc']

        split_files = [file for file in os.listdir(self.splits_loc) if file.endswith(".csv")]
        recent_split = sorted(split_files, key=lambda fn:os.path.getctime(os.path.join(self.splits_loc, fn)))[-1] # get most recent split
        split_file = pd.read_csv(os.path.join(self.splits_loc, recent_split))

        self.ids = [splitext(file)[0] for file in split_file[split_file['split'] == self.split]['tiles']]

        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)

    def preprocess(self, input_basemap_im, input_target_im, output_target_im):

        # turn basemap and target into tensors
        input_basemap_im = self.toTensor(input_basemap_im)*255 # the default is 0 to 255 turned into 0 to 1 -- override this since I'm doing my own normalizations
        input_target_im = self.toTensor(input_target_im) # no conversion. NA is -3.4e+38
        output_target_im = self.toTensor(output_target_im) # no conversion. NA is -3.4e+38

        input_target_im[input_target_im<=-3.4e+30] = 0      # turn invalid values to zero for model input
        output_target_im[output_target_im<=-3.4e+30] = float('nan') # -3.4e+38 to NaN 

        # normalize and generate the inputs and outputs

        # output
        output = (output_target_im - self.target_mean)/self.target_sd

        # input
        input_1 = (input_target_im - self.target_mean)/self.target_sd
        input_2 = (input_basemap_im[[0]] - self.basemap_mean1)/self.basemap_sd1
        input_3 = (input_basemap_im[[1]] - self.basemap_mean2)/self.basemap_sd2
        input_4 = (input_basemap_im[[2]] - self.basemap_mean3)/self.basemap_sd3
        input = torch.cat([input_1, input_2, input_3, input_4], dim=0)

        return input, output

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
        input_basemap_im = list(self.input_basemap.glob(name + '.*'))
        input_target_im = list(self.input_target.glob(name + '.*'))
        output_target_im = list(self.output_target.glob(name + '.*'))

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

        input, output = self.preprocess(input_basemap_im, input_target_im, output_target_im)

        return {
            'image': input,
            'label': output
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

    dataset = BasicDataset(cfg, 'test')
    len(dataset)
    dataset.__getitem__(1)