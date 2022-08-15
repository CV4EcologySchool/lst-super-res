import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

criterion = nn.MSELoss(reduction='none')


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    running_loss = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, label = batch['image'], batch['label']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            pred = net(image)

            mask = torch.isnan(label)
            label[mask] = -3.4e+30 # set missing values to any real number -- doesn't matter because it will be ignored in the loss. 
            loss = criterion(pred, label)
            loss[mask] = 0             # ignore pixel locations where target LST is NaN

            running_loss += loss.mean().item()    # the .item is to extract the value of the tensor with only one number in it

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return running_loss
    return running_loss/ num_val_batches
