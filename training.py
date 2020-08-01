
import warnings
warnings.filterwarnings("ignore")

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
import getopt

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from model import NVidia#nvidia model and the data loader
from model import RGBOpticalFlowDataset
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

def train(valid = False,test=False model_save=False):
    #    MODEL
    num_epoch = 10 #100 #90
    batch_size= 16

    model= NVidia().to(device=device)
    criterion = nn.MSELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)

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
        frames_to_rgb_flow("test",crop=crop)


    train_set = RGBOpticalFlowDataset("train.csv",ROOT)
    valid_set = RGBOpticalFlowDataset("valid.csv",ROOT)

    trainloader = DataLoader(train_set, batch_size= batch_size)
    validloader = DataLoader(valid_set, batch_size= batch_size)

    plotloss = []
    running_loss=0.0
    model.train()

    for epoch in range(num_epoch):
        for i, (images, speeds) in enumerate(trainloader):
            #if i==0:
            #if valid: eval(plotloss, validloader, trainloader, model, criterion)

            images = images.to(device=device,dtype=torch.float)
            speeds = speeds.to(device=device,dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(images).to(device=device)
            loss = criterion(outputs,speeds)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i%200==199:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1,running_loss/200))
                running_loss = 0.0


    if model_save:
        modeltar= "batch"+str(batch_size)+"_epoch"+str(num_epoch)+".tar"
        torch.save({'epoch':num_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_train_losses':plotloss}, os.path.join("/home/aras/Desktop", modeltar))


    if test==True:
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
    opts, args = getopt.getopt(argv,"",["valid=","test=","model_save="])
    opts = [(name[2:],val) for name, val in opts]
    train(**dict(opts))


if __name__ == "__name__":
    main(sys.argv[1:])









#
