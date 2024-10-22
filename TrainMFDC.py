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

Denoiser, optimizer = Load_model('./SSFD_G_NBnew20000')

##

Denoiser=SMFD_C(Denoiser).cuda()
loss = nn.L1Loss().cuda()

optimizer = optim.Adam(Denoiser.parameters(),lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001, amsgrad=False)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=500, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)




##
paths=['cl1']

##
trainset = Burstfolder('cl1lite', 0.1,8, MyRandomCrop(64),Randomnoise=False, loader = RGB_loader,loadram='cpu')
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,batch_size=8,drop_last=True)


##



trainburstserveur(Denoiser,paths, loss, optimizer, scheduler,"MFDCnew" , Nb_frames =8,batch_size=13, num_epochs = 5,nb_subepoch=100,save_every=1)




##



trainburstserveur2(Denoiser,trainloader, loss, optimizer, scheduler,"smallMFDCnew", num_epochs = 1000,save_every=100)













