#cd desktop/Burst-denoising


import torch.optim as optim
from torch.utils.data import DataLoader

from ImagesloaderRAM import *
from Burstload import *
from Trainfunction import *
from Savemodel import *
from Mycudatransformation import *
from MynnModule import *
from PSNR import *

import torchvision.transforms as T
import torchvision.utils as tu

import numpy as np
import pylab as plt

from time import time

dtype = torch.FloatTensor
gpu_dtype=torch.cuda.FloatTensor



"""
TO auto import modules
%load_ext autoreload
%autoreload 2

%reload_ext autoreload
"""






##
torch.cuda.empty_cache()


##

#Data_preprocessing=MyRandomCrop(300)
testset ,testloader=0,0
testset = Burstfolder('./8',0.2,8,Randomnoise=False, loader= RGB_loader,loadram='cpu')
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=1,num_workers=0)

##

Denoiser, optimizer , loss,loss_sfd,loss_mfd = Load_modelloss_mfd('./MFDCnewre27')

##

SFD ,optimizer,loss2=Load_modelloss('./SFD_C_NB15000')
##

Show_burst2(Denoiser,SFD, testloader,10,framerate=2, tour = 1 ,check=False)


##


plt.show()
plt.ion()

plt.figure(11)
plt.clf()
plt.plot(range(0,len(loss)),loss)

plt.figure(12)
plt.clf()
plt.plot(range(0,len(loss_sfd)),loss_sfd)
plt.plot(range(0,len(loss_mfd)),loss_mfd)

##

a=torch.cuda.memory_cached(device=None)










