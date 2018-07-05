#cd desktop/denoising


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

Denoiser, optimizer = Load_model('./SFD_G_NB300000')

##

Denoiser=MFD_G(Denoiser).cuda()
loss = nn.L1Loss().cuda()

optimizer = optim.Adam(Denoiser.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001, amsgrad=False)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5000, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)




##

Data_preprocessing=MyRandomCrop(64)                                        
trainset = Burstfolder('./test',0.1,8,Data_preprocessing,Randomnoise=False, loader= L_loader,loadram='cpu')
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=1,num_workers=0)

##



trainburst(Denoiser,trainloader, loss, optimizer, scheduler, Nb_frames =8, num_epochs = 1000,save_every=1000,loss_every=1)


##

Save_model(Denoiser,optimizer,'testmf45d')
##
torch.cuda.empty_cache()


##

Data_preprocessing=MyRandomCrop(400)    
testset ,testloader=0,0                                   
testset = Burstfolder('./testc',0.1,8,Data_preprocessing,Randomnoise=False, loader= RGB_loader,loadram='cpu')
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=1,num_workers=0)

##

Denoiser, optimizer = Load_model('./MFDC1')
##

Show_burst(Denoiser, testloader,5,chan=3,framerate=3,check=False)


##




a=torch.cuda.memory_cached(device=None)










