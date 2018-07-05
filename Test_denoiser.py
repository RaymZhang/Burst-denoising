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
import pylab as plt

from time import time

dtype = torch.FloatTensor
gpu_dtype=torch.cuda.FloatTensor


plt.show()
plt.ion()
"""
TO auto import modules
%load_ext autoreload
%autoreload 2
%reload_ext autoreload
"""



##
Denoiser, optimizer,loss= Load_modelloss('./25SFD_C_NB75000')

##
Denoiser2, optimizer,loss2= Load_modelloss('./SFD_C_B225000')

##☺
Denoiser3,optimizer, loss3 = Load_modelloss('./MFDC1')


##

Data_test=T.Compose([ 
                                        T.ToTensor()])  


testset = ImageFolderRAM('./CBSD68',0.1,Data_test,Randomnoise=False,loadram=False, loader=RGB_loader)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=1)


##


Show_result(Denoiser, testloader,0.1,check=False)


##

Show_result2(Denoiser,Denoiser2, testloader,0.1,check=False)





##
plt.show()
plt.ion()

##
plt.figure(5)
plt.clf()
plt.semilogy(range(0,len(loss2))[:10000:10],loss2[:10000:10])

plt.figure(6)
plt.clf()
plt.plot(range(0,len(loss))[:10000:10],loss[:10000:10])

##
plt.figure(6)
plt.clf()
plt.plot(range(0,len(loss2))[1::10],loss2[10000::10])


##
plt.figure(7)
plt.clf()
plt.plot(range(0,len(loss3)),loss3)






##




class MSFD_C(nn.Module):
    def __init__(self,MFD):
        
        super(MSFD_C, self).__init__()
        
        self.S1=MFD.S1
        self.S2=MFD.S2
        self.S3=MFD.S3
        self.S4=MFD.S4
        self.S5=MFD.S5
        self.S6=MFD.S6
        self.S7=MFD.S7
        self.S8=MFD.S8
        self.S9=MFD.S9
        
        
    def forward(self, x):
        out=self.S1(x)
        out=self.S2(out)
        out=self.S3(out)
        out=self.S4(out)
        out=self.S5(out)
        out=self.S6(out)
        out=self.S7(out)
        out=self.S8(out)
        out=self.S9(out)
        out += x
        return out



Denoiser4= MSFD_C(Denoiser3)











