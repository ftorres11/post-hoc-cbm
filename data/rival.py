# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa

# Torch imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random

# In package imports 

# Package imports
import pdb
import os
osp = os.path
osj = osp.join
import sys
import pickle
import numpy as np
from PIL import Image


# Classes
# ========================================================================
class AttrLoader(Dataset):
    def __init__(self, root_data, parent_data, transform=None,
                 return_name=False):
        self.root = root_data
        self.data = parent_data
        self.transform = transform
        self.return_name = return_name

    def __getitem__(self, idx):
        elem_dict = self.data[idx]
        path = elem_dict['img_path']
        label =  elem_dict['class_label']
        attrs = elem_dict['attribute_label']
        # Path processing given CUB data processed contains path info
        # from the original processing.
        if 'CUB' in path:
            path_split = path.strip('\n').split('/')
            cub_idx = path_split.index('CUB_200_2011')
            path = '/'.join(path_split[cub_idx+1::])
            fname = path_split[-1]
        img = Image.open(osj(self.root, path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.return_name:
            return img, label, torch.tensor(attrs), fname
        else:
            return img, label, torch.tensor(attrs)

    def __len__(self):
        return len(self.data)

class RIVALConceptDataset:
    def __init__(self, root, images, transform=None):
        self.images = images
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_concept_dicts(metadata):
    n_concepts = len(metadata[0]["attribute_label"])
    concept_info = {c: {1: [], 0: []} for c in range(n_concepts)}
    for idx, sample in enumerate(metadata):
        img, labels, attr = sample
        for c, label in enumerate(sample['attribute_label']):
            concept_info[c][label].append(sample[img])
    return concept_info
