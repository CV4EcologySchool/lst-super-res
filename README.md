# lst-super-res
This repository constitutes a U-Net model whose goal is to increase the resolution of a low resolution image with the help with a seperate high resolution input of the same resolution as the the desired output resolution. Concretely, this model was developed to increase the resolution of Land Surface Temperate (LST) images from 70m to 10m with the help of a 10m RGB basemap. The code for our U-Net was adapted from https://github.com/milesial/Pytorch-UNet. 

This code is highly flexible and, as with the U-Net implementation we borrow our basic structure from, takes any resonably sized image (try ~300-2000 pixels on each side). There are two inputs into the model: a basemap (in our case RGB), which should be at the resolution of the desired output, and a coarse target (in our case LST) which should be at the desired resolution of your original image you are hoping to increase the resolution of, but resaized to the same resolution as the basemap. The output which the model will be trained on should be the same size and resolution as the basemap input. 


## Installation instructions

1. Install [Conda](http://conda.io/)

2. Create environment and install requirements

```bash
conda create -n lst-super-res python=3.9 -y
conda activate lst-super-res
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt
```

_Note:_ depending on your setup you might have to install a different version of PyTorch (e.g., compiled against CUDA). See https://pytorch.org/get-started/locally/

3. Add data

The processed dataset for this particular project is currently not publicly available. However, one should manually add inputs and outputs and specify their paths in a configs/*.yaml file. Speficially, you will need to add paths to three folders:

- `input_basemap`: This folder is constituted of 8-bit, 3-band images of your basemap of choice (in our case RGB), all the same size and resolution (e.g. 672x672 pixels at 10m resolution)
- `input_target`: This folder is constituted of single band floating point images of your target (in our case LST) at a coarse resolution (e.g. 70m) but resampled to the same size and resolution as the basemap and the desired output. 
- `output_target`: This folder is constituted of your labels: single band floating point images of your target (in our case LST) at the desired improved resolution (e.g. 10m in our case.) 

_Note:_ Corresponding images from matching scenes should be named the same between folders, or you will get an error. 

The location of some metadata must also be included in your configs file:
- `splits_loc`: This the location of your file that determines how your dataset is to be split. It should contain a csv with the name of an image and whether it belongs to the "train", "val" or "test" set. The most recent file in this folder are used as your split. 
- `target_norm_loc`: This is a space delimited file that includes mean and sd columns with entries for all of your target input images (in our case, LST). The average across these are taken to normalize the inputs. 
- `basemap_norm_loc`: This is a space delimited file that includes mean (mean1, mean2, mean3) and sd (sd1, sd2, sd3) columns with entries for all of your input basemap images (in our case, RGB). The average across these are taken to normalize the inputs. 

- The metadata on the runs which includes information on their land cover type, `runs_metadata.csv` should be stored in a folder named "metadata" which is within your `data_root` as specified in your configs file. 

_Note:_ It is OK to have NA values in the input and output target, but not in your basemap. There is built-in functionality to ignore areas where there is no information: input NAs are set to 0 and output NAs are ignored when calculating the loss.

4. Split data

```bash
python3 code/split.py
```

## Reproduce results

1. Train

In the configs folder, create a *.yaml file for your experiment. See base.yaml as a example. 

```bash
python code/train.py --config configs/base.yaml
```

This will create a trained model which is saved at each epoch in the checkpoints folder

2. Predictions and validation

During training, weights and biases (wandb) is used to automatically generate visualizations of the training data and plot out the loss (MSE) of the training and validation sets. Wandb logs are generated and saved in the folder `code/wandb`. 

Generate predictions. These will be saved in the predictions folder of your experiment. If predictions are desired for another split, you can also specify 'test' or 'train'. 

```bash
python code/predict.py --config configs/base.yaml --split train
python code/predict.py --config configs/base.yaml --split val
```
Then, visualize and calcualte metrics for your predictions. These are also saved in your experiments folder. 

```bash
python code/predict_vis.py --config configs/base.yaml --split train
python code/predict_vis.py --config configs/base.yaml --split val
```

3. Test/inference

```bash
python code/predict.py --config configs/base.yaml --split test
python code/predict_vis.py --config configs/base.yaml --split test
```
