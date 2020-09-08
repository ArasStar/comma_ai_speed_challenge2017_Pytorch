'''
This script takes multiple raw data sequences gathered in 'kitti_dataset/clean_data' folder
sequences are downloaded from http://www.cvlibs.net/datasets/kitti/raw_data.php
as a label it takes forward velocity(m/s)(-fv-9th attribute in the dataformat) and convert that metric label to mph
save a csv file each row containing ['image_path', 'sequence_name', 'image_index', 'speed']

'''
import warnings
warnings.filterwarnings("ignore")
import os
import shutil
import pandas as pd
import cv2
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = "/home/aras/Desktop/commaAI"
CLEAN_KITTI_DATA_PATH = os.path.join(os.path.join(ROOT,'kitti_dataset',"clean_data"))
KITTI_TRAIN_IMGS= os.path.join(CLEAN_KITTI_DATA_PATH, 'train_imgs')

mps_to_mph_factor = 2.237
dataset_type='train'
seqs_list = os.listdir(KITTI_TRAIN_IMGS)
n_seqs = len(seqs_list)

meta_dict = {}
for i, file in enumerate(seqs_list):
    tqdm.write(f'{i+1}/{n_seqs} processing {file}')

    highres_imgs = os.path.join(KITTI_TRAIN_IMGS, file, 'image_03','data')
    highres_labels = os.path.join(KITTI_TRAIN_IMGS, file, 'oxts', 'data')
    seq_list = os.listdir(highres_labels)
    n_seq = len(seq_list)
    #iterating through sequence
    for sample_file_name in tqdm(seq_list,total=n_seq):
        idx = int(sample_file_name.split('.')[-2])
        img_path = os.path.join('kitti_dataset/clean_data/train_imgs',file,sample_file_name.split('.')[-2]+'.png')

        label_path =   os.path.join(highres_labels, sample_file_name)
        label_df = pd.read_csv(label_path,sep = " ", header=None)

        frame_speed = label_df[8].values[0] * mps_to_mph_factor
        meta_dict[file+'-'+str(idx)] = [img_path, file, idx, frame_speed]


meta_df = pd.DataFrame.from_dict(meta_dict, orient='index')
meta_df.columns = ['image_path', 'sequence_name', 'image_index', 'speed']

tqdm.write('writing meta to csv')
meta_df.to_csv(os.path.join(CLEAN_KITTI_DATA_PATH, dataset_type+'_meta.csv'), index=False)

print(meta_df.shape)
print(meta_df.head())
