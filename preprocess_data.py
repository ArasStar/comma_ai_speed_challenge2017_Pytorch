import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import skvideo.io
from tqdm import tqdm
import pandas as pd
import h5py
import sys
import os

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

def preprocess_image(image,kitti):
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
    if kitti:
        image_cropped = image[50:, 150:-150] # -> (380, 550, 3) #original
    else:
        image_cropped = image[100:440, :-90] # -> (380, 550, 3) #original

    image = cv2.resize(image_cropped, (220, 66), interpolation = cv2.INTER_AREA)

    return image

def preprocess_image_from_path(image_path, speed, bright_factor=None, train_mode=True, kitti=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if train_mode and bright_factor is not None:
        img = change_brightness(img, bright_factor)

    img = preprocess_image(img,kitti)

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

    ROOT = "/home/aras/Desktop/commaAI"
    train_y = list(pd.read_csv(os.path.join(ROOT, "raw", 'train.txt'), header=None, squeeze=True))
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
    lookup_df = pd.read_csv(dframe_loc)
    dframe = lookup_df[:-1]

    print('shuffling')
    trainN = int(len(dframe)*0.8)
    validN = len(dframe) - trainN
    datatype = ['train'] * trainN + ['valid']* validN
    assert(len(datatype)==len(dframe))

    datatype_col = pd.Series(datatype).sample(len(datatype),random_state=seed_val).values
    dframe['datatype'] = datatype_col
    dframe = dframe.sample(n=len(dframe),random_state=seed_val+2)

    print("--done--")

    return dframe ,lookup_df.set_index(['sequence_name','image_index']).sort_index()

def train_valid_split_kitti(dframe_loc, seed_val=1):
    """ shuffles and splits each sequence equally
    """
    print("---Reading.. Kitti")
    #group by shuffle and split each seq separetly concat them givem them test train shit do -1 from each for lookup_df
    lookup_df = pd.read_csv(dframe_loc).set_index(['sequence_name','image_index']).sort_index()
    dframe = pd.DataFrame()

    print('shuffling Kitti')
    for g_id, g in lookup_df.groupby('sequence_name'):

        g_df = g[:-1]
        trainN = int(len(g_df)*0.8)
        validN = len(g_df) - trainN
        datatype = ['train'] * trainN + ['valid']* validN
        assert(len(datatype)==len(g_df))
        datatype_col = pd.Series(datatype).sample(len(datatype),random_state=seed_val).values
        g_df['datatype'] = datatype_col
        dframe = pd.concat([dframe,g_df])

    dframe = dframe.sample(n=len(dframe), random_state=seed_val+2)
    print("--done-- Kitti")

    return dframe.reset_index() ,lookup_df

def windowAvg(dframe,datatype):

    window_size = 25
    print(datatype+' data shape: ', dframe.shape, dframe.columns)

    #dframe = pd.read_csv(os.path.join(__file__[:__file__.rfind('/')],csv_file))
    for seq_name, seq_df in dframe.groupby('sequence_name'):
        seq_df = seq_df.sort_values(['image_index'])
        seq_df['smooth_predicted_speed'] = seq_df['predicted_speed'].rolling(window_size, center=True).median()

        seq_df['smooth_predicted_speed'] = seq_df.apply(lambda x: x['predicted_speed'] if np.isnan(x['smooth_predicted_speed'])
                                                                                             else x['smooth_predicted_speed'],axis=1)

        fig, ax = plt.subplots(figsize=(20,10))
        plt.xlabel('image_index (or time since start)')
        plt.ylabel('speed')

        if datatype == 'test':
            print("saving the .txt in to the same folder that the script runs")
            output_file = seq_df['smooth_predicted_speed']
            output_file.to_csv(os.path.join(__file__[:__file__.rfind('/')], datatype + '_output.txt'),index=False)
            title = 'Predictions on '+ datatype.upper() +' data'
            plt.title(title)
            ax.set_title(title)

            plt.plot(seq_df[['image_index']], seq_df[['predicted_speed']], 'bx')
            plt.plot(seq_df[['image_index']], seq_df[['smooth_predicted_speed']], 'g.')
            plt.legend(['predicted speed', 'smooth predicted speed'], loc='upper right')

        elif datatype == 'valid':
            mse_ = "MSE:" + '{:.3f}'.format(mean_squared_error(seq_df["speed"], seq_df["predicted_speed"]))
            mse_smooth = "MSE(smooth):" + '{:.3f}'.format(mean_squared_error(seq_df["speed"], seq_df["smooth_predicted_speed"]))
            title = 'Predictions on '+datatype.upper()+' data - sequence :' + seq_name
            plt.title(title)

            plt.plot(seq_df[['image_index']], seq_df[['speed']], 'go', label= 'validation speed')
            plt.plot(seq_df[['image_index']], seq_df[['predicted_speed']], 'bx', label ='predicted speed')
            plt.plot([],[],label=mse_)
            plt.plot([],[],label=mse_smooth)
            plt.plot([],[],label=str(len(seq_df))+' samples')
            plt.legend( loc='upper right')

        elif datatype == 'train':
            mse_ = "MSE:" + '{:.3f}'.format(mean_squared_error(seq_df["speed"], seq_df["predicted_speed"]))
            mse_smooth = "MSE(smooth):" + '{:.3f}'.format(mean_squared_error(seq_df["speed"], seq_df["smooth_predicted_speed"]))
            title = 'Predictions on '+datatype.upper()+' data - sequence :' + seq_name
            plt.title(title)
            ax.set_title(title)

            plt.plot(seq_df[['image_index']], seq_df[['predicted_speed']], 'bx', label='predicted_speed')
            plt.plot(seq_df[['image_index']], seq_df[['smooth_predicted_speed']], 'g.',label= "25 window average")
            plt.plot(seq_df[['image_index']], seq_df[['speed']], 'r.',label="ground truth")
            plt.plot([],[],label=mse_)
            plt.plot([],[],label=mse_smooth)
            plt.plot([],[],label=str(len(seq_df))+' samples')
            plt.legend( loc='upper right')

        plt.draw()
        plt.pause(0.05)
        print(title)
        title_path = os.path.join(__file__[:__file__.rfind('/')],'plots',seq_name)
        if not os.path.exists(title_path):
            os.mkdir(title_path)
        plt.savefig(os.path.join(title_path, title+'.png'))
        #Tried to close the figures here but it didn't work
        #plt.close()
