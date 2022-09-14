#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0,"..")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage, skimage.io
import pprint
import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
import torchxrayvision as xrv

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('-img_path', type=str,default="1.jpg")
parser.add_argument('-weights', type=str,default="all")
parser.add_argument('-feats', default=False, help='', action='store_true')
parser.add_argument('-cuda', default=False, help='', action='store_true')
cfg = parser.parse_args()

img = skimage.io.imread(cfg.img_path)
img = xrv.datasets.normalize(img)

# Check that images are 2D arrays
if len(img.shape) > 2:
    img = img[:, :, 0]
if len(img.shape) < 2:
    print("error, dimension lower than 2 for image")

# Add color channel
img = img[None, :, :]

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])
img = transform(img)
# d_chex = xrv.datasets.CheX_Dataset(
#     imgpath="path to CheXpert-v1.0-small",
#     csvpath="path to CheXpert-v1.0-small/train.csv",
#     transform=transform)

# 224x224 models
# model = xrv.models.DenseNet(weights=cfg.weights)
# model = xrv.models.DenseNet(weights="densenet121-res224-all")
model = xrv.models.DenseNet(weights="densenet121-res224-chex")
# model = xrv.models.DenseNet(weights="densenet121-res224-nih")
# # 512x512 models
# model = xrv.models.ResNet(weights="resnet50-res512-all")

# # DenseNet121 from JF Healthcare for the CheXpert competition
# model = xrv.baseline_models.jfhealthcare.DenseNet() 

# # Official Stanford CheXpert model
# model = xrv.baseline_models.chexpert.DenseNet()



output = {}
with torch.no_grad():
    img = torch.from_numpy(img).unsqueeze(0)
    if cfg.cuda:
        img = img.cuda()
        model = model.cuda()
        
    if cfg.feats:
        feats = model.features(img)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        output["feats"] = list(feats.cpu().detach().numpy().reshape(-1))

    preds = model(img).cpu()
    output["preds"] = dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))
    
if cfg.feats:
    print(output)
else:
    pprint.pprint(output)
    

# python process_image.py -img_path padchest_sample/sample/101103270798497222826083823719046670601_jw1fu2.png -feats -cuda