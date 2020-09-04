
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
#from sklearn.utils import shuffle
#import skvideo.io
from tqdm import tqdm
import pandas as pd
import h5py
import sys
import getopt

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
#from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as transforms
import torch.optim as optim
from model import NVidia#nvidia model and the data loader
from model import RGBOpticalFlowDataset
from model import customloader


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("CUDA(GPU) FOUND!!\n")
else:
    print("cuda not available -- terminating..")
    sys.exit()

# constants
ROOT = "/home/aras/Desktop/commaAI"
MODEL_DIR = os.path.join(ROOT,'mycode','models')
CLEAN_DATA_PATH = os.path.join(ROOT,"speed_challenge_2017/clean_data")
CLEAN_IMGS_TRAIN = os.path.join(CLEAN_DATA_PATH ,'train_imgs')
CLEAN_IMGS_TEST = os.path.join(CLEAN_DATA_PATH ,'test_imgs')
#TESTING
def make_predictions(model,data_csv):
    print("loading data for prediction")
    testloader, shape = customloader(CLEAN_DATA_PATH,ROOT, data_csv, batch_size =16, datatype="test")

    if isinstance(model,str):
        model_dict = torch.load(model)
        nvidia = NVidia(image_size=shape)
        nvidia.load_state_dict(model_dict["model_state_dict"])
        model = nvidia
    elif not isinstance(model,nn.Module):
        print("model needs to be either a path(str) or a nn.Module - parameter error terminating...")
        sys.exit()

    res = None
    #model.cpu()
    model.to(device=device)
    model.eval()
    tqdm.write("making predictions...")
    for _,(imgs,_) in tqdm(enumerate(testloader),total=len(testloader)):
        imgs = imgs.to(device=device)
        outputs = model(imgs)
        if res is None:
            res = outputs.cpu().detach().numpy()
        else:
            res = np.concatenate((res,outputs.cpu().detach().numpy()), axis=0)

    res = np.concatenate((res,[res[-1]]), axis=0)
    res = pd.DataFrame(res,columns=['predicted_speed'])
    res['image_index'] = pd.read_csv(os.path.join(CLEAN_DATA_PATH,data_csv))['image_index']
    res.to_csv(os.path.join(ROOT,'mycode','output.csv'))


def eval(model,criterion, validloader):
    model.eval()
    running_loss = 0.0
    count=0
    for images, speeds in enumerate(validloader):
            images = images.to(device=device,dtype=torch.float)
            speeds = speeds.to(device=device,dtype=torch.float).view(-1,1)
            outputs = model(images)
            running_loss += criterion(outputs,speeds).item()
            count += len(images)
    model.train()
    avg_loss = running_loss/count
    return avgloss

def train(valid=False, test=False, plot=False ,save_model=False, num_epoch=15, batch_size=16, interval=1):
    # Setting data & model
    trainloader, validloader, shapeImage = customloader(CLEAN_DATA_PATH,ROOT, "train_meta.csv", batch_size=16, datatype="train")

    model= NVidia(image_size=shapeImage).to(device=device)

    criterion = nn.MSELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)

    plotloss = []
    valid_loss = []
    n_iter = len(trainloader)
    running_loss=0.0

    print("# of trainable params: ", sum(p.numel() for p in model.parameters()))
    print('N of iteration', n_iter*num_epoch)

    if plot:
        title = f'batchsize_{batch_size}__epoch_{num_epoch}'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        curve, = plt.plot([])
        #ax.set_ylim(bottom=0, top=100)
        #ax.set_xlim(left=-100, right=num_epoch*n_iter)

    start_time = time.time()
    epoch_start = start_time
    model.train()
    for epoch in range(num_epoch):
        if valid:
            val_loss = eval(model,criterion, validloader)
            valid_loss.append(val_loss)

        for i, (images, speeds) in enumerate(trainloader):

            images = images.to(device=device, dtype=torch.float)
            speeds = speeds.to(device=device,dtype=torch.float).view(-1,1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,speeds)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (epoch*n_iter + i )%interval==(interval-1):
                avgloss = running_loss/interval
                print(f'epoch:{epoch+1}/{num_epoch}, {i+1}/{n_iter} loss: {avgloss}')
                running_loss = 0.0
                if plot:
                    curve.set_ydata(np.append(curve.get_ydata(),avgloss))
                    curve.set_xdata(np.append(curve.get_xdata(),(epoch*n_iter)+i+1))
                    ax.relim()
                    ax.autoscale_view(True,True,True)
                    ax.set_ylim(bottom=0)
                    plt.draw()
                    plt.pause(0.0001)

        print(f'epoch {epoch+1} finished in {time.time()-epoch_start}')
        epoch_start = time.time()

    print(f'Training of {num_epoch} epoch finsihed in {time.time()-start_time} seconds -- approx. {(time.time()-start_time)/60.0} minutes  ')


    #SAVE MODEL & PLOT
    if save_model:
        print('saving..')
        if plot: plt.savefig(title)
        modeltar= "batch"+str(batch_size)+"_epoch"+str(num_epoch)+".tar"
        torch.save({'epoch':num_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_train_losses':plotloss}, os.path.join(MODEL_DIR, modeltar))

    if test:
        print("making prediction on test set")
        make_predictions(model,'test_meta.csv')

def main(argv):
    def str2val(args):
        for idx,[name,strval] in enumerate(args):
            if name in ["valid","test","plot","save_model"]:
                if strval in ["True","true", "1","t"]:
                    args[idx]=(name,True)
                elif strval in ["False","false", "0","f"]:
                    args[idx]=(name,False)
                else:
                    print("ERRRRROR ARGUMENT PARSING:", name,strval)

            elif name in ["num_epoch","batch_size","interval"]:
                if strval.isdigit():
                    args[idx]=(name,int(strval))
                else:
                    print("ERRRRROR ARGUMENT PARSING:", name,strval)
            else:
                print("ERRRRROR ARGUMENT PARSING:", name,strval)

    opts, args = getopt.getopt(argv,"",["valid=","test=","plot=","save_model=","num_epoch=","batch_size=","interval="])
    opts = [(name[2:],val) for name, val in opts]
    if opts: str2val(opts)
    train(**dict(opts))


if __name__ == "__main__":
    main(sys.argv[1:])
