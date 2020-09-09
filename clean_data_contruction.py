import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv
import skvideo.io
from tqdm import tqdm
import ffmpeg
#from tqdm.notebook import tqdm # for notebook
#%matplotlib inline

# constants
DATA_PATH = "/home/aras/Desktop/commaAI/speed_challenge_2017/data"
TRAIN_VIDEO = os.path.join(DATA_PATH, 'train.mp4')
TEST_VIDEO = os.path.join(DATA_PATH, 'test.mp4')
CLEAN_DATA_PATH = "/home/aras/Desktop/commaAI/speed_challenge_2017/clean_data"
CLEAN_IMGS_TRAIN = os.path.join(CLEAN_DATA_PATH, 'train_imgs')
CLEAN_IMGS_TEST = os.path.join(CLEAN_DATA_PATH, 'test_imgs')

train_frames = 20400
test_frames = 10798

if not os.path.exists(CLEAN_IMGS_TRAIN):
    os.makedirs(CLEAN_IMGS_TRAIN)

if not os.path.exists(CLEAN_IMGS_TEST):
    os.makedirs(CLEAN_IMGS_TEST)

#labels
train_y = list(pd.read_csv(os.path.join(DATA_PATH, 'train.txt'), header=None, squeeze=True))

assert(len(train_y)==train_frames)

def dataset_constructor(video_loc, img_folder, tot_frames, dataset_type):
    meta_dict = {}

    tqdm.write('reading in video file...')
    cap = skvideo.io.vreader(video_loc)
    print(type(cap))

    tqdm.write('constructing dataset...')
    for idx, frame in enumerate(tqdm(cap)):
        img_path = os.path.join(img_folder, str(idx)+'.jpg')
        frame_speed = float('NaN') if dataset_type == 'test' else train_y[idx]
        meta_dict[idx] = [img_path, idx, frame_speed,'comma_ai']
        skvideo.io.vwrite(img_path, frame)

    meta_df = pd.DataFrame.from_dict(meta_dict, orient='index')
    meta_df.columns = ['image_path', 'image_index', 'speed','sequence_name']

    tqdm.write('writing meta to csv')
    meta_df.to_csv(os.path.join(CLEAN_DATA_PATH, dataset_type+'_meta.csv'), index_=False)

    return "done dataset_constructor"


dataset_constructor(TRAIN_VIDEO, CLEAN_IMGS_TRAIN, train_frames, 'train')

train_meta = pd.read_csv(os.path.join(CLEAN_DATA_PATH, 'train_meta.csv'))
assert(train_meta.shape[0] == train_frames)
#assert(train_meta.shape[1] == 3)
print("pass")
train_meta.head()


dataset_constructor(TEST_VIDEO, CLEAN_IMGS_TEST, test_frames, 'test')









#
