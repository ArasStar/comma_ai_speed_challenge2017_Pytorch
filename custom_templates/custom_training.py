
import warnings
warnings.filterwarnings("ignore")
import os
import sys

sys.path.insert(1,"/home/aras/Desktop/commaAI/mycode")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
import skvideo.io
from tqdm import tqdm
import pandas as pd
import h5py
import getopt
from math import ceil

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
#import visdom
#from model import NVidia#nvidia model and the data loader
#from model import RGBOpticalFlowDataset

from model_custom import NVidia#nvidia model and the data loader
from model_custom import CustomDataset

from preprocess_data import video_to_frames
from preprocess_data import frames_to_rgb_flow



if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("CUDA(GPU) FOUND!!\n")
else:
    print("cuda not available -- terminating..")
    sys.exit()

# constants
ROOT = "/home/aras/Desktop/commaAI/speed_challenge_2017"
MODEL_DIR = "/home/aras/Desktop/commaAI/models"

# Load DATASET Train, Valid and Test
if not os.path.exists(os.path.join(ROOT,"clean","train.csv")):
    print("no clean data found videos to frames working now\n")
    video_to_frames(os.path.join(ROOT,"raw","train.mp4"),os.path.join(ROOT,"clean","train"),"train") # video_path and img folder
    video_to_frames(os.path.join(ROOT,"raw","test.mp4"),os.path.join(ROOT,"clean","test"),"test")

if not os.path.exists(os.path.join(ROOT,"rgb_flow","train.csv")):
    print("no clean rgbflow found, frame pairs to rgb_flow working now, also shuffle at the end\n")
    plot = False
    crop = None #flow
    frames_to_rgb_flow("train", crop=crop)
    frames_to_rgb_flow("test", crop=crop)


def eval(plot_loss,validloader,trainloader,model,criterion):
    val_train_loss = [None,None]
    model.eval()
    for idx,dataloader in enumerate([validloader,trainloader]):
        running_loss = 0
        count=0
        for  i, (x, y) in enumerate(dataloader):
            x = x.to(device=device,dtype=torch.float)
            y = y.to(device=device,dtype=torch.float)
            outputs = model(x).to(device=device)
            running_loss += criterion(outputs,y).item()
            count += 1
        avg_loss = running_loss/count
        val_train_loss[idx] = avg_loss

    plot_loss.append(val_train_loss)
    model.train()

def train(valid=False, test=False, plot=False ,save_model=False, num_epoch=85, batch_size=16, interval=200):

    #    MODEL
    train_set = CustomDataset("paired_clean_train.csv",ROOT)
    valid_set = CustomDataset("paired_clean_valid.csv",ROOT)
    trainloader = DataLoader(train_set, batch_size= batch_size,num_workers=8)
    validloader = DataLoader(valid_set, batch_size= batch_size)

    shape = tuple(train_set[0][0].shape)
    model= NVidia(image_size=shape).to(device=device)
    criterion = nn.MSELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)

    plotloss = []
    n_iter = len(trainloader)
    running_loss=0.0

    print("# of trainable params: ", sum(p.numel() for p in model.parameters()))
    print('N of iteration', n_iter*num_epoch)

    model.train()
    if plot:
        title =f'batchsize_{batch_size}__epoch_{num_epoch}'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        curve, = plt.plot([])
        #ax.set_ylim(bottom=0, top=100)
        #ax.set_xlim(left=-100, right=num_epoch*n_iter)

    for epoch in range(num_epoch):
        for i, (images, speeds) in enumerate(trainloader):
            #if i==0 and valid: eval(plotloss, validloader, trainloader, model, criterion)

            images = images.to(device=device,dtype=torch.float)
            speeds = speeds.to(device=device,dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(images).to(device=device)
            loss = criterion(outputs,speeds)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (epoch*n_iter + i )%interval==(interval-1):
                avgloss = running_loss/interval
                print(f'epoch:{epoch+1}, {i+1}/{n_iter} loss: {avgloss}')
                running_loss = 0.0
                if plot:
                    curve.set_ydata(np.append(curve.get_ydata(),avgloss))
                    curve.set_xdata(np.append(curve.get_xdata(),(epoch*n_iter)+i+1))
                    ax.relim()
                    ax.autoscale_view(True,True,True)
                    ax.set_ylim(bottom=0)
                    plt.draw()
                    plt.pause(0.0001)

    # saving plot
    if False and plot:
        plt.savefig(title)

    #SAVE MODEL
    if save_model:
        modeltar= "batch"+str(batch_size)+"_epoch"+str(num_epoch)+".tar"
        torch.save({'epoch':num_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_train_losses':plotloss}, os.path.join("/home/aras/Desktop", modeltar))

    #TESTING
    if test:
        print("testing")
        test_set = RGBOpticalFlowDataset("test.csv",ROOT)
        testloader = DataLoader(test_set, batch_size= batch_size)
        res = None
        model.eval()
        model.cpu()
        for  i, (x, y) in enumerate(testloader):
            x = x.cpu()
            y = y.cpu()
            outputs = model(x).cpu()
            if res is None:
                res = outputs
            else:
                res = torch.cat((res.cpu(),outputs.cpu())).cpu()

        res = torch.cat((res,res[0].view(1,1)))

        print(res.shape)
        np.savetxt("/home/aras/Desktop/test.txt",res.detach().numpy(),delimiter=', ')


def main(argv):
    def str2val(args):
        for idx,[name,strval] in enumerate(args):
            if name in ["valid","test","plot","model_save"]:
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

    opts, args = getopt.getopt(argv,"",["valid=","test=","plot=","model_save=","num_epoch=","batch_size=","interval="])
    opts = [(name[2:],val) for name, val in opts]
    if opts: str2val(opts)
    train(**dict(opts))


if __name__ == "__main__":
    main(sys.argv[1:])




#
