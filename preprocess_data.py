import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
import skvideo.io
from tqdm import tqdm
import pandas as pd
import h5py
import sys
import os

# constants
ROOT = "/home/aras/Desktop/commaAI/speed_challenge_2017"

def change_brightness(image, bright_factor):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor

    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb


def preprocess_image(image, crop=None):
    """
    preprocesses the image

    input: image (480 (y), 640 (x), 3) RGB
    output: image (shape is (220, 66, 3) as RGB)

    This stuff is performed on my validation data and my training data
    Process:
             1) Cropping out black spots
             3) resize to (220, 66, 3) if not done so already from perspective transform
    """
    # Crop out sky (top) (100px) and black right part (-90px)
    if crop=="more":
        image_cropped = image[140:400, :-90] # -> (380, 550, 3)
    else:
        image_cropped = image[100:440, :-90] # -> (380, 550, 3) #original

    image = cv2.resize(image_cropped, (220, 66), interpolation = cv2.INTER_AREA)

    return image

def preprocess_image_from_path(image_path, speed, bright_factor=None,crop=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if bright_factor is not None:
        img = change_brightness(img, bright_factor)

    img = preprocess_image(img,crop=crop)
    return img, speed

def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros((66, 220, 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]

    # Flow Parameters
    #     flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)

    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    return rgb_flow

def video_to_frames(video_path, img_folder, dataset_type):
    '''
    takes the frames out of the video and creates a csv_file:
    '''
    train_y = list(pd.read_csv(os.path.join(ROOT,"raw", 'train.txt'), header=None, squeeze=True))
    meta_dict = {}

    #download video
    tqdm.write('reading in video file...')
    cap = skvideo.io.FFmpegReader(video_path)
    tot_frames = cap.getShape()[0]
    cap.close()
    cap = skvideo.io.vreader(video_path)


    tqdm.write('constructing dataset...')
    for idx, frame in enumerate(tqdm(cap,total=tot_frames)):
        img_path = os.path.join(img_folder, str(idx)+'.jpg')
        frame_speed = float('NaN') if dataset_type == 'test' else train_y[idx]
        meta_dict[idx] = [img_path, idx, frame_speed]
        skvideo.io.vwrite(img_path, frame)

    meta_df = pd.DataFrame.from_dict(meta_dict, orient='index')
    meta_df.columns = ['image_path', 'image_index', 'speed']

    tqdm.write('writing meta to csv')
    meta_df.to_csv(os.path.join(ROOT,"clean", dataset_type+'.csv'), index=False)

    return "done dataset_constructor--videos to frames"


def train_valid_split(dframe_loc, seed_val=1):
    """ shuffles and splits with the same ratio as Jovsa
    """
    print("---Reading..")
    dframe = pd.read_csv(dframe_loc)
    
    print("---Shuffling..")
    shuffled = dframe.iloc[:-1].sample(n=len(dframe)-1 ,random_state=seed_val)

    print('---Creating pairs succesive frames')
    paireddf = pd.DataFrame()
    for i in tqdm(range(len(shuffled))):
        idx1 = shuffled.iloc[i].image_index
        idx2 = idx1 + 1

        row1 = dframe.iloc[[idx1]].reset_index()
        row2 = dframe.iloc[[idx2]].reset_index()

        paired_frames = [paireddf, row1, row2]
        paireddf = pd.concat(paired_frames, axis = 0, join = 'outer', ignore_index=False)


    print("len of paired df  ",len(paireddf))

    print("---Spliting...")
    split_idx = int(len(shuffled)*0.8)*2

    train_data = paireddf.iloc[:split_idx]
    valid_data = paireddf.iloc[split_idx:]
    print("--done--")

    return train_data, valid_data


def generate_training_data(data, batch_size = 32):
    image_batch = np.zeros((batch_size, 66, 220, 3)) # nvidia input params
    label_batch = np.zeros((batch_size))
    idx = 1
    while True:

        for i in range(batch_size):
            if idx > len(data)-2:
                idx = 1

            # Generate a random bright factor to apply to both image
            bright_factor = 0.2 + np.random.uniform()

            row_now = data.iloc[[idx]].reset_index()
            row_prev = data.iloc[[idx - 1]].reset_index()
            row_next = data.iloc[[idx + 1]].reset_index()

            # Find the 3 respective times to determine frame order (current -> next)

            time_now = row_now['image_index'].values[0]
            time_prev = row_prev['image_index'].values[0]
            time_next = row_next['image_index'].values[0]

            if abs(time_now - time_prev) == 1 and time_now > time_prev:
                row1 = row_prev
                row2 = row_now

            elif abs(time_next - time_now) == 1 and time_next > time_now:
                row1 = row_now
                row2 = row_next
            else:
                print('Error generating row-generate training data')

            x1, y1 = preprocess_image_from_path(row1['image_path'].values[0],
                                                row1['speed'].values[0],
                                               bright_factor)

            # preprocess another image
            x2, y2 = preprocess_image_from_path(row2['image_path'].values[0],
                                                row2['speed'].values[0],
                                               bright_factor)

            # compute optical flow send in images as RGB
            rgb_diff = opticalFlowDense(x1, x2)

            # calculate mean speed
            y = np.mean([y1, y2])

            image_batch[i] = rgb_diff
            label_batch[i] = y

            idx += 2

        #print('image_batch', image_batch.shape, ' label_batch', label_batch)
        # Shuffle the pairs before they get fed into the network
        yield shuffle(image_batch, label_batch)
