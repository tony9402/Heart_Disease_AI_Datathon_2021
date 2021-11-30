import torch
import torch.nn.functional as F

def bce(pred, target):
    return F.binary_cross_entropy(pred, target)

def _dice_metric(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    score = (2 * intersection + 1.) / (union + 1.)
    return score

def dice_metric(pred, target):
    metric = _dice_metric(pred, target)
    return metric.sum()

# Using Image Segmentation
def dice_loss(pred, target):
    dice = _dice_metric(pred, target)
    loss = 1.0 - dice
    return loss, dice

def bce_dice_loss(pred, target):
    loss, dice = dice_loss(pred, target)
    loss = loss.sum()
    loss += bce(pred, target).sum()
    return loss

def load_loss(loss_type):
    if loss_type == "bce":
        return bce
    elif loss_type == "bce_dice_loss":
        return bce_dice_loss
    elif loss_type == "dice_loss":
        return dice_loss
    else:
        raise KeyError(f"Not Exist {loss_type} loss function")