# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa
# Pending updates - deletions to current repository

# Torch imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, distributed
import random

# In package imports

# Package imports
import os
import pdb
import sys
import pickle
import numpy as np
from PIL import Image

# Classes
# ========================================================================
class Adjacent_Loader(Dataset):
    def __init__(self, data, concepts, adj_mat):
        self.data = data
        self.adj = adj_mat[concepts:, :concepts]
        self.concepts = concepts

    def __getitem__(self, idx):
        img, label = self.data[idx]
        attr = self.adj[label, :]
        return img, label, attr

    def __len__(self):
        return len(self.labels)

class fMNISTConceptDataset:
    def __init__(self, images, idxs, transform=None):
        self.images = images[idxs]
        self.transform = transform
    def  __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if self.transform:
            image = self.transform(image)
        return image

def get_concept_dicts(data):
    n_concepts = data.concepts
    concept_info = {c: {1: [], 0: []} for c in range(n_concepts)}
    for idx, sample in enumerate(data):
        img, labels, attr = sample
        for c, label in enumerate(attr):
            #print(c)
            concept_info[c][label].append(idx)
    return concept_info

