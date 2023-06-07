'''
    This script contains miscellaneous util functions that are declared in other .py files.

    2022 Anna Boser
'''

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

def normalize_pretrain(target_im, target_mean, target_sd, mean_for_nans=True):

    # retrieve the mean and sds from the basemap
    target_mean = target_mean
    target_sd = target_sd

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

def unnormalize_image(normalized_image, basemap_norms, target_norms, n_bands = 3):

    assert n_bands == 3, \
        'Currently only basemaps with three bands are supported.'
    
    # retrieve the mean and sds from the basemap
    target_mean = target_norms['mean']
    target_sd = target_norms['sd']

    # normalize
    target_im = (normalized_image[[0]]*target_sd) + target_mean

    # retrieve the mean and sds from the basemap
    basemap_mean1 = basemap_norms['mean1'] 
    basemap_mean2 = basemap_norms['mean2']
    basemap_mean3 = basemap_norms['mean3']
    basemap_sd1 = basemap_norms['sd1']
    basemap_sd2 = basemap_norms['sd2']
    basemap_sd3 = basemap_norms['sd3']

    # normalize
    input_1 = (normalized_image[[1]]*basemap_sd1) + basemap_mean1
    input_2 = (normalized_image[[2]]*basemap_sd2) + basemap_mean2
    input_3 = (normalized_image[[3]]*basemap_sd3) + basemap_mean3

    return (torch.cat([target_im,input_1,input_2,input_3], dim=0)).int()

def randomize(r,g,b, seed = 1234):
    # Set seed
    random.seed(seed)
    # Declare number of r/g/b bands that will be iterated on
    rgb_num = np.random.randint(2,10)
    rgb_list = []
    # Generate list of r/g/b bands to iterate on
    for i in range(rgb_num):
        rgb_list.append(np.random.choice(['r','g','b']))
    op_num = rgb_num - 1
    ops = {'+':operator.add,
        '-':operator.sub,
        '*':operator.mul,
        '/':operator.truediv}
    op_list = []
    counter = 0
    # Perform correct operation according to proper r/g/b combination
    for i in range(op_num):
        # Create list of randomized operations
        op_list.append(np.random.choice(list(ops.keys())))
        # Perform first operation between two bands
        if counter == 0:
            if (rgb_list[i] == 'r') & (rgb_list[i+1] == 'r'):
                rgb_list[i+1] = ops.get(op_list[i])(r,r)
            elif (rgb_list[i] == 'r') & (rgb_list[i+1] == 'g'):
                rgb_list[i+1] = ops.get(op_list[i])(r,g)
            elif (rgb_list[i] == 'r') & (rgb_list[i+1] == 'b'):
                rgb_list[i+1] = ops.get(op_list[i])(r,b)
            elif (rgb_list[i] == 'g') & (rgb_list[i+1] == 'r'):
                rgb_list[i+1] = ops.get(op_list[i])(g,r)
            elif (rgb_list[i] == 'g') & (rgb_list[i+1] == 'g'):
                rgb_list[i+1] = ops.get(op_list[i])(g,g)
            elif (rgb_list[i] == 'g') & (rgb_list[i+1] == 'b'):
                rgb_list[i+1] = ops.get(op_list[i])(g,b)
            elif (rgb_list[i] == 'b') & (rgb_list[i+1] == 'r'):
                rgb_list[i+1] = ops.get(op_list[i])(b,r)
            elif (rgb_list[i] == 'b') & (rgb_list[i+1] == 'g'):
                rgb_list[i+1] = ops.get(op_list[i])(b,g)
            elif (rgb_list[i] == 'b') & (rgb_list[i+1] == 'b'):
                rgb_list[i+1] = ops.get(op_list[i])(b,b)
            rgb_list[i+1] += 1 # Ensure no division by 0 occurs
            counter += 1
        # Perform next operation
        else:
            # Normalize data if values start to explode
            if np.max(np.abs(rgb_list[i])) > 5000:
                rgb_list[i+1] = ( rgb_list[i] - np.mean(rgb_list[i]) )/ np.std(rgb_list[i]) 
                continue
            if rgb_list[i+1] == 'r':
                rgb_list[i+1] = ops.get(op_list[i])(rgb_list[i],r)
            elif rgb_list[i+1] == 'g':
                rgb_list[i+1] = ops.get(op_list[i])(rgb_list[i],g)
            elif rgb_list[i+1] == 'b':
                rgb_list[i+1] = ops.get(op_list[i])(rgb_list[i],b) 
    # Store mean and standard deviation of randomized output
    random_mean = np.mean(rgb_list[-1])
    random_sd = np.std(rgb_list[-1])
    # Print mean and standard deviation
    print('Mean:', random_mean, 'SD:', random_sd, flush = True)
    # Convert randomized output to Image object
    randomized_image = Image.fromarray(rgb_list[-1])
    return randomized_image, random_mean, random_sd # Returns randomized output including its mean and standard deviation