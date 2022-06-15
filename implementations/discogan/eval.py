# calculate iou of a set of predicted masks and ground truth masks

import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import cv2
import sklearn.metrics as skmetrics


def denormalize(tensor):
    tensor = tensor[0]
    tensor = tensor * 0.5 + 0.5
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def calculate_area(pred: np.ndarray, label: np.ndarray, num_classes: int, ignore_index=255):
    """
    Calculate intersect, prediction and label area
    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = np.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = np.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape,
                                                           label.shape))
    pred_area = []
    label_area = []
    intersect_area = []
    mask = label != ignore_index

    for i in range(num_classes):
        pred_i = np.logical_and(pred == i, mask)
        label_i = label == i
        intersect_i = np.logical_and(pred_i, label_i)
        pred_area.append(np.sum(pred_i.astype(int)))
        label_area.append(np.sum(label_i.astype(int)))
        intersect_area.append(np.sum(intersect_i.astype(int)))

    pred_area = np.array(pred_area)
    label_area = np.array(label_area)
    intersect_area = np.array(intersect_area)

    return intersect_area, pred_area, label_area


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.
    Args:
        intersect_area (ndarray): The intersection area of prediction and ground truth on all classes.
        pred_area (ndarray): The prediction area on all classes.
        label_area (ndarray): The ground truth area on all classes.
    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def iou_score(pred, target):
    # pred = torch.squeeze(pred)
    # target = torch.squeeze(target)
    pred = denormalize(pred)
    target = denormalize(target)

    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    pred = cv2.threshold(pred, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    target = cv2.threshold(target, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    cv2.imwrite("test.png", pred*255)
    cv2.imwrite("test2.png", target*255)

    intersect_area, pred_area, label_area = calculate_area(pred, target, 2)
    class_iou, iou = mean_iou(intersect_area, pred_area, label_area)

    return iou


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
adversarial_loss = torch.nn.MSELoss()
cycle_loss = torch.nn.L1Loss()
pixelwise_loss = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorUNet(input_shape)

if cuda:
    G_AB = G_AB.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)

# Input tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)


# ----------
#  Training
# ----------

ious = []

for i, batch in tqdm(enumerate(dataloader)):

    # Model inputs
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))

    G_AB.eval()

    # GAN loss
    fake_B = G_AB(real_A)

    # calculate iou between real_B and fake_B
    iou = iou_score(real_B, fake_B)
    ious.append(iou)

ious = np.array(ious)
print("mean iou: %.4f" % np.mean(ious))
print("mean iou(>0.1): %.4f" % np.mean(ious[ious > 0.1]))
