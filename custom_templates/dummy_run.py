import warnings
warnings.filterwarnings("ignore")

import torch
from model_custom import NVidia

import matplotlib.pyplot as plt
import numpy as np
import time

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("CUDA(GPU) FOUND!!\n")
else:
    print("cuda not available -- terminating..")
    sys.exit()

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, H2, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        h_relu = self.linear2(h_relu).clamp(min=0)
        y_pred = self.linear3(h_relu)
        return y_pred

'''
x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')

for phase in np.linspace(0, 10*np.pi, 100):

    line1.set_ydata(np.sin(0.5 * x + phase))
    fig.canvas.draw()
    #plt.show()
    plt.pause(0.0001)
'''

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, H2, D_out = 640, 2000, 1000,10000, 1

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in).to(device=device)
y = torch.randn(N, D_out).to(device=device)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, H2, D_out).to(device=device)


N = 1

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, 3,66,220).to(device=device)
y = torch.randn(N, 1).to(device=device)

# Construct our model by instantiating the class defined above
model = NVidia().to(device=device)


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.

criterion = torch.nn.MSELoss(reduction='sum').to(device=device)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("# of trainable params: ", sum(p.numel() for p in model.parameters()))

fig = plt.figure()
ax = fig.add_subplot(111)
curve, = plt.plot([])
ax.set_ylim(bottom=0, top=100)
ax.set_xlim(left=-100, right=40000)

for idx,t in enumerate(range(40000)):
    # Forward pass: Compute predicted y by passing x to the model
    #print(model.linear2.weight)
    y_pred = model(x).to(device=device)
    break
    # Compute and print loss
    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 100== 0:
        print(t,"------", loss.item())

        curve.set_ydata(np.append(curve.get_ydata(),loss.item()))
        curve.set_xdata(np.append(curve.get_xdata(),idx+1))

        #ax.relim()
        #ax.autoscale_view(True,True,True)
        #ax.set_ylim(bottom=0)

        plt.draw()
        plt.pause(0.0001)

        #curve.set_ydata(curve.get_ydata(),loss.item)

        #print('difference')
        #print(model.out.weight-prev)
        #print('==============================================')







    #
