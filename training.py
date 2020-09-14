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
from preprocess_data import windowAvg

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("CUDA(GPU) FOUND!!\n")
else:
    print("cuda not available -- terminating..")
    sys.exit()

# constants
project_folder_name='mycode'
ROOT = "/home/aras/Desktop/commaAI"
MODEL_DIR = os.path.join(ROOT,project_folder_name,'models')
CLEAN_DATA_PATH = os.path.join(ROOT,"speed_challenge_2017/clean_data")
CLEAN_IMGS_TRAIN = os.path.join(CLEAN_DATA_PATH ,'train_imgs')
CLEAN_IMGS_TEST = os.path.join(CLEAN_DATA_PATH ,'test_imgs')

#TESTING
def make_predictions(model, datatype, dataloader=None, kitti=False):
    print("loading "+ (datatype if dataloader is None else 'valid')+" data for prediction")
    #setting up the data you want to do predictions on
    if dataloader is None:
        index = pd.read_csv(os.path.join(CLEAN_DATA_PATH, datatype + '_meta.csv'))['image_index']
        loader, shape = customloader(CLEAN_DATA_PATH, ROOT, datatype + '_meta.csv', batch_size =1, datatype="test", kitti=kitti)
    else:
        loader = dataloader

    #get either a Nvidia model object or path to the model.tar
    if isinstance(model,str):
        model_dict = torch.load(model)
        nvidia = NVidia()
        nvidia.load_state_dict(model_dict["model_state_dict"])
        model = nvidia
    elif not isinstance(model,nn.Module):
        print("model needs to be either a path(str) or a nn.Module - parameter error terminating...")
        sys.exit()

    res = {}
    model.to(device=device)
    model.eval()
    tqdm.write("making predictions for " + datatype + ' set ...')
    for row_i,(img,speed, idx, seq_name) in tqdm(enumerate(loader),total=len(loader)):
        img = img.to(device=device)
        predicted_speed = model(img).cpu().detach()
        res[row_i] = [predicted_speed.item(), speed.item(), idx.item(), seq_name[0]]

    res = pd.DataFrame.from_dict(res, orient="index", columns = ['predicted_speed', 'speed', 'image_index', 'sequence_name'] )

    if dataloader is None:
        for seq_name, g in res.groupby('sequence_name'):
            last_row = g.loc[g['image_index'] == len(g)-1]
            last_row['image_index'] += 1
            res = pd.concat([res,last_row])

    windowAvg(res,datatype)

def eval(model, criterion, validloader):
    model.eval()
    running_loss = 0.0
    count = len(validloader)
    for _, (images, speeds, _, _) in enumerate(validloader):
            images = images.to(device=device,dtype=torch.float)
            speeds = speeds.to(device=device,dtype=torch.float).view(-1,1)
            outputs = model(images)
            running_loss += criterion(outputs,speeds).item()
    model.train()
    avg_loss = running_loss/count
    return avg_loss

def train(valid=False, test=False, plot=False ,save_model=False, num_epoch=15, batch_size=16, interval=1, steps_per_epoch=None, kitti=False):
    # Setting data & model
    trainloader, validloader, shapeImage = customloader(CLEAN_DATA_PATH, ROOT, "train_meta.csv", batch_size=16, datatype="train", kitti=kitti)

    model= NVidia(image_size=shapeImage).to(device=device)
    criterion = nn.MSELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)

    n_iter = len(trainloader) if steps_per_epoch is None else steps_per_epoch
    print("# of trainable params: ", sum(p.numel() for p in model.parameters()))
    print('N of iteration', n_iter*num_epoch)

    if plot:
        title = f'batchsize_{batch_size}__epoch_{num_epoch}{"_kitti" if kitti else ""}'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        curve_train, = plt.plot([],label = "training loss")
        curve_valid, = plt.plot([], label = "validation loss")
        plt.legend(loc='upper right')
        ax.set_ylim(bottom=0, top=50)
        #ax.set_xlim(left=-100, right=num_epoch*n_iter)

    start_time = time.time()
    epoch_start = start_time
    model.train()
    for epoch in range(num_epoch):
        if valid:
            val_loss = eval(model,criterion, validloader)
            print(f'VALIDATION loss before starting epoch {epoch+1} is: {val_loss}')
            if plot:
                curve_valid.set_ydata(np.append(curve_valid.get_ydata(),val_loss))
                curve_valid.set_xdata(np.append(curve_valid.get_xdata(),(epoch*n_iter)+1))

        for i, (images, speeds, _,_) in enumerate(trainloader):
            images = images.to(device=device, dtype=torch.float)
            speeds = speeds.to(device=device,dtype=torch.float).view(-1,1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,speeds)
            loss.backward()
            optimizer.step()

            if   (i == 0 and epoch == 0) or (interval!="epoch" and i %interval==(interval-1)) or (interval=='epoch' and i == len(trainloader)-1):
                print(f'epoch:{epoch+1}/{num_epoch}, {i+1}/{n_iter} loss: {loss.item()}')

                if plot:
                    curve_train.set_ydata(np.append(curve_train.get_ydata(),loss.item()))
                    curve_train.set_xdata(np.append(curve_train.get_xdata(),(epoch*n_iter)+i+1))
                    ax.relim()
                    ax.autoscale_view(True,True,True)
                    plt.pause(0.0001)

            if steps_per_epoch is not None and i > steps_per_epoch: break

        print(f'epoch {epoch+1} finished in {time.time()-epoch_start}')



        epoch_start = time.time()

    print(f'Training of {num_epoch} epoch finsihed in {time.time()-start_time} seconds -- approx. {(time.time()-start_time)/60.0} minutes')

    if valid:
        val_loss = eval(model,criterion,validloader)
        print("final validation score =", val_loss)
        if plot:
            curve_valid.set_ydata(np.append(curve_valid.get_ydata(),val_loss))
            curve_valid.set_xdata(np.append(curve_valid.get_xdata(),((num_epoch)*n_iter)+1))
            ax.relim()
            ax.autoscale_view(True,True,True)
            plt.pause(0.001)

    #SAVE MODEL & PLOT
    if save_model:
        print('saving..')

        modeltar= "batch"+str(batch_size)+"_epoch"+str(num_epoch)+".tar"
        torch.save({'epoch':num_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(MODEL_DIR, modeltar))

    if plot:
        if save_model:
            plt.savefig('plots/'+title)
        #tried to close the figure here but couldn't find a way
    if test:
        print("making predictions test/train(full)/valid set ")
        make_predictions(model,'valid',dataloader=validloader,kitti=kitti)
        make_predictions(model,'test')
        make_predictions(model,'train',kitti=kitti)


def main(argv):
    def str2val(args):
        for idx,[name,strval] in enumerate(args):
            if name in ["valid","test","plot","save_model","kitti","interval"]:
                if strval in ["True","true", "1","t"]:
                    args[idx]=(name,True)
                elif strval in ["False","false", "0","f"]:
                    args[idx]=(name,False)
                elif name =="interval" and strval=="epoch":
                    args[idx]=(name,"epoch")
                else:
                    print("ERRRRROR ARGUMENT PARSING:", name,strval)

            elif name in ["num_epoch","batch_size","interval","steps_per_epoch"]:
                if strval.isdigit():
                    args[idx]=(name,int(strval))
                else:
                    print("ERRRRROR ARGUMENT PARSING:", name,strval)
            else:
                print("ERRRRROR ARGUMENT PARSING:", name,strval)

    opts, args = getopt.getopt(argv,"",["valid=","test=","plot=","save_model=","num_epoch=","batch_size=","interval=","steps_per_epoch=", "kitti="])
    opts = [(name[2:],val) for name, val in opts]
    if opts: str2val(opts)
    train(**dict(opts))

if __name__ == "__main__":
    main(sys.argv[1:])






#
