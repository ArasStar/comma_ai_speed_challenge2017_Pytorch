from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from math import floor
import matplotlib.pyplot as plt
import sys

if "/home/aras/Desktop/commaAI/mycode" not in sys.path:
    sys.path.insert(1,"/home/aras/Desktop/commaAI/mycode")

from preprocess_data import pair_to_rgbflow


# constants
ROOT = "/home/aras/Desktop/commaAI/speed_challenge_2017"

#train_frames = 20400
#test_frames = 10798

###### DATASET #######
class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir=ROOT,
                transform = transforms.Compose([transforms.ToTensor()])):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """

        self.clean_paired = pd.read_csv(os.path.join(root_dir,"clean_data",csv_file))
        self.root_dir = root_dir
        self.transforms = transform

    def __len__(self):
        return len(self.clean_paired)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, speed = pair_to_rgbflow(self.clean_paired.iloc[idx])

        if self.transforms:
            image = self.transforms(image)

        return image,speed


#######MODEL########
class NVidia(nn.Module):
    def __init__(self,image_size=(3,66, 220), init_weights=True):
        super(NVidia, self).__init__()
        def shape_out(h_in,stride,kernel_size, padding=0, dilation=1):
            '''padding is 0 and dilation is 1'''
            return floor(((h_in + 2*padding - dilation*(kernel_size-1)-1) / stride) +1)

        def find_shape( h_, w_, convs):
            for kernel_size,stride in convs:
                h_ = shape_out(h_,stride,kernel_size)
                w_ = shape_out(w_,stride,kernel_size)
            return h_,w_

        ch , h, w = image_size
        self.layer_types =[nn.Conv2d,nn.Linear]
        self.conv1 = nn.Conv2d(ch, 24, (5,5) ,stride=(2,2))
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(24, 36, (5,5), stride=(2,2))
        self.conv3 = nn.Conv2d(36, 48, (5,5), stride=(2,2))
        self.drop = nn.Dropout(p=0.5)
        self.conv4 = nn.Conv2d(48, 64, (3,3), stride=(1,1))
        self.conv5 = nn.Conv2d(64, 64, (3,3), stride=(1,1))
        self.flatten = nn.Flatten()

        new_h, new_w = find_shape(h,w, [(5,2),(5,2),(5,2),(3,1),(3,1)])

        #self.extra = nn.Linear(new_h*new_w*64,new_h*new_w*64)
        self.fc1 = nn.Linear(new_h*new_w*64, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        self.out = nn.Linear(10, 1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        ### DO CUSTOM NORMALIZATION  HERE

        x = x /127.5 -1
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.drop(self.elu(self.conv3(x)))
        x = self.elu(self.conv4(x))
        x = self.elu(self.flatten(self.conv5(x)))

        #x = self.elu(self.extra(x))

        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))

        x = self.out(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if any(isinstance(m, type) for type in self.layer_types):
                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
