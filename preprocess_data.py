'''
PREPROCESS --> OPTICAL FLOW --> SHUFFLE
'''
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




def read_clean(dataset_type):
    print("reading clean "+dataset_type+ "dataset to_csv")
    return pd.read_csv(os.path.join(ROOT,"clean", dataset_type+'.csv')) #train2_meta

def pairup_data(df_data, dataset_type):
    '''creates a csv that each row has consecutive frames idx_(i),idx_(i+1)'''
    print("creating dataset csv with paired frames")
    df_data_current = df_data.iloc[:-1]
    df_data_next = df_data.iloc[1:].reset_index().drop(["index"], axis = 1)
    df_paired = df_data_current.merge(df_data_next,left_index=True, right_index=True)
    df_paired.to_csv(os.path.join(ROOT,"clean" 'paired_clean_'+dataset_type+'.csv'), index=False)
    print( "done pairing frames, shape is:",df_paired.shape)
    return df_paired


def create_data(df_paired, dataset_type, crop=None):
    ''' Preprocess pairs that calculate optical flow'''
    print("processing pairs for rgb_flow")
    meta_dict = {}
    for idx, pair in tqdm(df_paired.iterrows(),total=len(df_paired)):
        i1,i2 = pair["image_index_x"], pair["image_index_y"]
        x1_path ,x2_path = pair["image_path_x"],pair["image_path_y"]
        assert(i2-i1==1)
        img_path = os.path.join(ROOT,"rgb_flow",dataset_type,str(i1)+"_"+str(i2)+".jpg")

        bright_factor = 0.2 + np.random.uniform()
        x1,y1 = preprocess_image_from_path(x1_path, pair["speed_x"],bright_factor, crop)
        x2,y2 = preprocess_image_from_path(x2_path, pair["speed_y"],bright_factor, crop)
        rgb_flow = opticalFlowDense(x1,x2)
        speed = np.mean([y1,y2])

        meta_dict[idx] = [img_path, idx, speed]
        skvideo.io.vwrite(img_path, rgb_flow)


    meta_df = pd.DataFrame.from_dict(meta_dict, orient='index')
    meta_df.columns = ['image_path', 'image_index', 'speed']
    meta_df.to_csv(os.path.join(ROOT,"rgb_flow", dataset_type+'.csv'), index=False)

    print("finished preprocessing and calculated optical flow of pairs, shape is:", meta_df.shape)
    return meta_df


def shuffle_and_split(df_data, seed=1, split=[0.8,0.2]):
    print("shuffling and spliting to train and valid")
    #shuffle
    df_shuffled = df_data.sample(n=len(df_data), random_state=seed)

    if split == False:#no split
        train_data.to_csv(os.path.join(ROOT,"rgb_flow","train.csv"))
    else:
        assert(sum(split)==1.0)
        split_idx = int(len(df_data)*split[0])
        train_data, valid_data = df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:]
        train_data.to_csv(os.path.join(ROOT,"rgb_flow","train.csv"))
        valid_data.to_csv(os.path.join(ROOT,"rgb_flow","valid.csv"))


def frames_to_rgb_flow(dataset_type, split= [0.8,0.2], crop=None):
    print("creating rgbflow for "+ dataset_type)
    df_meta = read_clean(dataset_type)
    df_paired = pairup_data(df_meta,dataset_type)
    df_preprocessed = create_data(df_paired, dataset_type,crop=crop)
    if dataset_type == "train":
        shuffle_and_split(df_preprocessed,split=split)


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


def main(dataset_type):
    print("creating for "+dataset_type)
    path = path_dict[dataset_type]
    df_meta = read_clean(dataset_type)
    df_paired = pairup_data(df_meta,dataset_type)
    df_preprocessed= create_data(df_paired, path, dataset_type)
    if dataset_type == "train":
        shuffle_and_split(df_preprocessed)


if __name__ == "__main__":
    if len(sys.argv)>2:
        print("error in passing arg' you need to pass only one argument")
    elif len(sys.argv)==2:
        if not(sys.argv[1] == "train" or sys.argv[1] == "test"):
            print("wrong input:",sys.argv[1])
        else:
            main(sys.argv[1])
    else:
        main("train")




def trial_test():
    print("TRIALS")
    pic=10200
    bright_factor = None
    clean_pic_path = os.path.join(CLEAN_IMGS_TRAIN,str(pic)+".jpg")
    #clean_pic_path = os.path.join(CLEAN_IMGS_TEST,str(pic)+".jpg")

    img = cv2.imread(clean_pic_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img)

    if bright_factor is not None:
        img = change_brightness(img, bright_factor)
    img1 = preprocess_image(img)


    plt.figure()
    plt.imshow(img1)


    clean_pic_path = os.path.join(CLEAN_IMGS_TRAIN,str(pic+1)+".jpg")
    #clean_pic_path = os.path.join(CLEAN_IMGS_TEST,str(pic)+".jpg")

    img = cv2.imread(clean_pic_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img)

    if bright_factor is not None:
        img = change_brightness(img, bright_factor)
    img2 = preprocess_image(img)

    plt.figure()
    plt.imshow(img2)


    rgb_flow = opticalFlowDense(img1,img2)

    plt.figure()
    plt.imshow(rgb_flow)



    plt.show()
    plt.close()
