from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from math import floor
import matplotlib.pyplot as plt
from preprocess_data import train_valid_split, train_valid_split_kitti, preprocess_image_from_path, opticalFlowDense

ROOT = "/home/aras/Desktop/commaAI"
CLEAN_KITTI_DATA_PATH = os.path.join(os.path.join(ROOT,'kitti_dataset',"clean_data"))

###### DATASET COMMAI DATASET #######
class RGBOpticalFlowDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self,datatype, root_dir, dframe, lookup_df=None,kitti=False, transform = transforms.Compose([transforms.ToTensor()])):
        """
        Args:
            dframe (dataframe): dataframe .
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.dframe = dframe[dframe['datatype']==datatype]
        self.lookup_df = lookup_df
        self.root_dir = root_dir
        self.transforms = transform
        self.train_mode = datatype =='train'
        self.kitti = kitti

    def __len__(self):
        return len(self.dframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        '''
        id1 = self.dframe.index[idx]
        id2 = id1 + 1
        row1 = self.lookup_df.iloc[[id1]]
        row2 = self.lookup_df.iloc[[id2]]
        '''
        row1 = self.dframe.iloc[idx]
        seq_name, id1 = row1[['sequence_name', 'image_index']]
        id2 = id1 + 1
        row2 = self.lookup_df.loc[(seq_name, id2)]

        bright_factor = 0.2 + np.random.uniform()
        x1, y1 = preprocess_image_from_path(os.path.join(self.root_dir,row1['image_path']), row1['speed'],
                                            bright_factor=bright_factor,train_mode=self.train_mode,kitti=self.kitti)
        # preprocess another image
        x2, y2 = preprocess_image_from_path(os.path.join(self.root_dir,row2['image_path']), row2['speed'],
                                            bright_factor=bright_factor,train_mode=self.train_mode, kitti=self.kitti)
        # compute optical flow send in images as RGB
        rgb_diff = opticalFlowDense(x1, x2)

        # calculate mean speed
        speed = np.mean([y1, y2])

        if  self.transforms:
            rgb_diff = self.transforms(rgb_diff)

        return rgb_diff, speed, id1, seq_name


######DATALOADER Function#######
def customloader(rootcsv, rootD, csv_file, batch_size, datatype, kitti=False):

    if datatype == 'train':
        dframe, lookup_df = train_valid_split(os.path.join(rootcsv,csv_file))
        train_set = RGBOpticalFlowDataset('train', rootD, dframe, lookup_df)
        valid_set = RGBOpticalFlowDataset('valid', rootD, dframe, lookup_df)
        shape = tuple(train_set[0][0].shape)

        if kitti: #if kitti is mixed than we combine datasets
            dframe_kitti, lookup_df_kitti = train_valid_split_kitti(os.path.join(CLEAN_KITTI_DATA_PATH,'train_meta.csv'))
            train_set_kitti = RGBOpticalFlowDataset('train', rootD, dframe_kitti, lookup_df_kitti, kitti=True)
            valid_set_kitti = RGBOpticalFlowDataset('valid', rootD, dframe_kitti, lookup_df_kitti, kitti=True)
            assert(shape == tuple(train_set_kitti[0][0].shape) )

            train_set = ConcatDataset([train_set,train_set_kitti,train_set_kitti])
            valid_set = ConcatDataset([valid_set,valid_set_kitti,valid_set_kitti])

        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory= True, num_workers=8)#, collate_fn=collate_fn)
        validloader = DataLoader(valid_set, batch_size=1, pin_memory=True, num_workers=8)

        return trainloader, validloader ,shape

    else:
        test_lookup_df = pd.read_csv(os.path.join(rootcsv, csv_file)).set_index(['sequence_name','image_index']).sort_index()
        test_dframe = test_lookup_df.iloc[:-1].reset_index()
        test_dframe['datatype'] = 'test'

        test_set = RGBOpticalFlowDataset('test', rootD, test_dframe, test_lookup_df)
        shape = tuple(test_set[0][0].shape)

        if kitti:
            lookup_df_kitti = pd.read_csv(os.path.join(CLEAN_KITTI_DATA_PATH,'train_meta.csv')).set_index(['sequence_name','image_index']).sort_index()
            test_dframe_kitti = pd.concat([g[:-1] for g_id, g in lookup_df_kitti.groupby('sequence_name') ]).reset_index()
            test_dframe_kitti['datatype'] = 'test'

            test_set_kitti = RGBOpticalFlowDataset('test', rootD, test_dframe_kitti, lookup_df_kitti)
            assert(shape == tuple(test_set_kitti[0][0].shape) )
            test_set = ConcatDataset([test_set,test_set_kitti,test_set_kitti])

        testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=True, num_workers=8)

        return testloader, shape


#######MODEL########
class NVidia(nn.Module):
    def __init__(self,image_size=(3,66, 220), init_weights=True):
        super(NVidia, self).__init__()
        def shape_out(h_in,stride,kernel_size, padding=0):
            '''padding is 0 and dilation is 1'''
            return floor(((h_in - (kernel_size-1)-1) / stride) +1)

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
                        nn.init.zeros_(m.bias)
