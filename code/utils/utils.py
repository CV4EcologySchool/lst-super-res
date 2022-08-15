import matplotlib.pyplot as plt
import torch


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def normalize_target(target_im, target_norms, mean_for_nans=True):

    # retrieve the mean and sds from the basemap
    target_mean = target_norms['mean']
    target_sd = target_norms['sd']

    target_im[target_im<=-3.4e+30] = float('nan')      # -3.4e+38 to NaN

    # normalize
    target_im = (target_im - target_mean)/target_sd

    if mean_for_nans==True:
        target_im[torch.isnan(target_im)] = 0
    
    return target_im

def unnormalize_target(target_im, target_norms):

    # retrieve the mean and sds from the basemap
    target_mean = target_norms['mean']
    target_sd = target_norms['sd']

    # normalize
    target_im = (target_im*target_sd) + target_mean

    
    return target_im


def normalize_basemap(basemap_im, basemap_norms, n_bands=3):

    assert n_bands == 3, \
        'Currently only basemaps with three bands are supported.'

    # retrieve the mean and sds from the basemap
    basemap_mean1 = basemap_norms['mean1'] 
    basemap_mean2 = basemap_norms['mean2']
    basemap_mean3 = basemap_norms['mean3']
    basemap_sd1 = basemap_norms['sd1']
    basemap_sd2 = basemap_norms['sd2']
    basemap_sd3 = basemap_norms['sd3']

    # normalize
    input_1 = (basemap_im[[0]] - basemap_mean1)/basemap_sd1
    input_2 = (basemap_im[[1]] - basemap_mean2)/basemap_sd2
    input_3 = (basemap_im[[2]] - basemap_mean3)/basemap_sd3

    return input_1, input_2, input_3

def unnormalize_basemap(basemap_im, basemap_norms, n_bands=3):

    assert n_bands == 3, \
        'Currently only basemaps with three bands are supported.'

    # retrieve the mean and sds from the basemap
    basemap_mean1 = basemap_norms['mean1'] 
    basemap_mean2 = basemap_norms['mean2']
    basemap_mean3 = basemap_norms['mean3']
    basemap_sd1 = basemap_norms['sd1']
    basemap_sd2 = basemap_norms['sd2']
    basemap_sd3 = basemap_norms['sd3']

    # normalize
    input_1 = (basemap_im[[0]]*basemap_sd1) + basemap_mean1
    input_2 = (basemap_im[[1]]*basemap_sd2) + basemap_mean2
    input_3 = (basemap_im[[2]]*basemap_sd3) + basemap_mean3

    return (torch.cat([input_1, input_2, input_3], dim=0)).int()