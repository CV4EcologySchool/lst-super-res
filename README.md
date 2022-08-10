# lst-super-res
Increase the resolution of Land Surface Temperate (LST) images.

My theoretical file structure: 

code -- run in AWS EC2
- data_processing
  - tile
  - aggregate

data -- in AWS S3
- raw
  - LST
  - RGB
- tiles
  - LST
  - RGB
- aggregated
  - input_LST
  - output_LST

## Installation instructions

1. Install [Conda](http://conda.io/)

2. Create environment and install requirements

```bash
conda create -n lst-super-res python=3.9 -y
conda activate lst-super-res
pip install -r requirements.txt
```

_Note:_ depending on your setup you might have to install a different version of PyTorch (e.g., compiled against CUDA). See https://pytorch.org/get-started/locally/

3. Download dataset

**NOTE:** Requires the [azcopy CLI](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) to be installed and set up on your machine.

```bash
sh scripts/download_dataset.sh 
```

This downloads the [CCT20](https://lila.science/datasets/caltech-camera-traps) subset to the `datasets/CaltechCT` folder.


## Reproduce results

1. Train

```bash
python ct_classifier/train.py --config configs/exp_resnet18.yaml
```

2. Test/inference

@CV4Ecology participants: Up to you to figure that one out. :)

See also ct_classifier/visualize_predictions.ipynb
