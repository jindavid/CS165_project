"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""


import torch
import numpy as np
import gc
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)

#Complex multiplication
def compl_mul2d(a, b):
    op = partial(torch.einsum, "bctq,dctq->bdtq")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modefunctions, fourier):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modefunctions = modefunctions
        self.fourier = fourier

        self.scale = (1 / (in_channels * out_channels))
        if self.fourier:
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        else:
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        if self.fourier:
            x_ft = torch.rfft(x, 2, normalized=True, onesided=True)
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            #Return to physical space
            x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=(x.size(-2), x.size(-1)))
        else:
            POD = torch.tensordot(self.modefunctions[:,:,:self.modes1], x, dims = ([0, 1], [2, 3]))
            #Apply model parameters to POD modes. 
            weights = torch.einsum('ijk->kij',self.weights1)  
            POD_out = torch.einsum('ija,iak->ijk',POD,weights)
            #Return to physical space
            x = torch.tensordot(POD_out, self.modefunctions[:,:,:self.modes1], dims = ([0], [2]))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width, modefunctions, fourier):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.modefunctions = modefunctions
        self.fourier = fourier
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2, self.modefunctions, self.fourier)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2, self.modefunctions, self.fourier)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2, self.modefunctions, self.fourier)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2, self.modefunctions, self.fourier)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width, modefunctions, fourier):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes, width, modefunctions, fourier)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


################################################################
# configs
################################################################
#TRAIN_PATH = 'data/NavierStokes_V1e-5_N1200_T20.mat'
#TEST_PATH = 'data/NavierStokes_V1e-5_N1200_T20.mat'
TRAIN_PATH = 'data/Vortex_dynamics_64_64_train_2.mat'
TEST_PATH = 'data/Vortex_dynamics_64_64_test.mat'

ntrain = 1296
ntest = 256

modes = 10
width = 20
fourier = 1

batch_size = 8
batch_size2 = batch_size

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

path = 'ns_fourier_2d_rnn_V10000_T20_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

print('Will save model to:', path_model)

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64
T_in = 10
T = 2
step = 1

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]
print(train_a.size())
print(train_u.size())
mean = reader.read_field('Meantensor')
mean = mean[np.newaxis,:,:,np.newaxis]
modefunctions = reader.read_field('Modetensor')
print(mean[np.newaxis,:,:,np.newaxis].size())

train_a =train_a - mean;
train_u = train_u - mean;

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

test_a =test_a - mean;
test_u =test_u - mean;

modefunctions = modefunctions[:,:,:modes].to(device='cuda')

assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

# pad the location (x,y)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])

train_a = torch.cat((gridx.repeat([ntrain,1,1,1]), gridy.repeat([ntrain,1,1,1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,1]), gridy.repeat([ntest,1,1,1]), test_a), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')
print(test_a.size())
print(test_u.size())
################################################################
# training and evaluation
################################################################

model = Net2d(modes, width, modefunctions, fourier).cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

print(model.count_params())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


myloss = LpLoss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:-2], im,
                            gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        # l2_full.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:-2], im,
                                gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)


            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)
    
torch.save(model, path_model)


# test
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
print(type(test_loader))
pred = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    test_l2 = 0
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        t1 = default_timer()
        out = model(x)
        t2 = default_timer()
        pred[index] = out
        loss = myloss(out.view(1, -1), y[:,:,:,0].view(1, -1)).item()
        test_l2 += loss
        print(index, loss)
        index = index + 1
print(test_l2/ntest)
print(t2-t1)

path = 'eval_fourier'
scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy(), 'u': test_u.cpu().numpy()})
# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
#      for x, y in test_loader:
#          test_l2 = 0;
#          x, y = x.cuda(), y.cuda()

#          out = model(x)
#          #out = y_normalizer.decode(out)
#          pred[index] = out

#          test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#          print(index, test_l2)
#          index = index + 1

# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})


