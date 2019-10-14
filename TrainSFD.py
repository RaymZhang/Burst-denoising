#cd desktop/denoising

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ImagesloaderRAM import *
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

##




Denoiser = SFD_G().cuda()
loss = nn.L1Loss().cuda()
optimizer = optim.Adam(Denoiser.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001, amsgrad=False)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5000, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

Data_preprocessing=T.Compose([ T.RandomCrop(64),
                                        T.ToTensor()])



trainset = ImageFolderRAM('./BSD200G',0.1,Data_preprocessing,Randomnoise=False, loader= L_loader,loadram='cpu')
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64,num_workers=0)




##

trainserver(Denoiser,trainloader, loss, optimizer, scheduler, num_epochs = 500000,save_every=25000,loss_every=3,filename='SFD_G_NB')