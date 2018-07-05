import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
dtype = torch.FloatTensor
gpu_dtype=torch.cuda.FloatTensor

gpu_long=torch.cuda.LongTensor



##


"""
Perform a random square crop of size 'size' into a array of shape (T,C,H,W) on the last 2 dimentions.
"""
class MyRandomCrop(object):
    
    def __init__(self, size):
        self.size = size
      
    def __call__(self, img):
        a,c,h,w = img.shape
        th,tw = np.random.randint(0,h-self.size),np.random.randint(0,w-self.size)
        return(img[:,:,th:th+self.size,tw:tw+self.size])
        
    
        

    
        
        
        
