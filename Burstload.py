import torch.utils.data as data

from PIL import Image

import torchvision.transforms as T
import os
import os.path
import torch
import numpy as np
##
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    




    
    
## Functions to load a color image
def pil_RGB_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    
    
def accimage_RGB_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_RGB_loader(path)


def RGB_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_RGB_loader(path)
 
 
 
## Function to load a grey image
def pil_L_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
    
def accimage_L_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_L_loader(path)
        
def L_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_L_loader(path)
            
    


"""
Create a liste of burst of shape (T,C,H,W) where T is Nb_frames, the number of frame in one burst

"""

def make_dataset(dir,Nb_frames):
    images = []
    dir = os.path.expanduser(dir)
    for scene in sorted(os.listdir(dir)):
        d = os.path.join(dir,scene)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            i=0
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = path
                    images.append(item)
                    i+=1
                    if i==Nb_frames :
                        break 
    return images




def make_dataset_burst_ram(dir,Nb_frames,loader=RGB_loader):
    path=make_dataset(dir,Nb_frames)
    bursts=[]
    Transform=T.ToTensor()
    
    A=[]
    c=0
    
    for i in path:
        img=Transform(loader(i))
        A.append(img)
        c+=1
        if c%Nb_frames == 0:
            bursts.append(torch.stack(A))
            A=[]
        
    return(bursts)   



    

class Burstfolder(data.Dataset):
   
# images must be in a folder dir
    """
    Args:
        root (string): Root directory path.
        Data_preprocessing (callable, optional): A function/transform that takes float tensor of shape (T,C,H,W) where T is the number of frame and returns a transformed version.
        sigma : the level of noise we are going to add
        Randomnoise : if True a random noise in [0,sigma]
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        imgs (list):(image path) 
    """

    def __init__(self, root,sigma=0,Nb_frames=8, Data_preprocessing=None, Randomnoise=False, loader=RGB_loader,loadram='cpu'):
        
        
       
        if loadram == 'cpu' :
            imgs = make_dataset_burst_ram(root,Nb_frames,loader)
            
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.loadram=loadram
        self.Randomnoise=Randomnoise
        self.sigma = sigma
        self.root = root
        self.imgs = imgs
        self.Data_preprocessing = Data_preprocessing
        #self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue 
        """
        
        if  self.loadram == 'cpu' :
            img=self.imgs[index]
            
            
        if self.Data_preprocessing is not None:            
            img = self.Data_preprocessing(img)
            groundtrue=img.clone() 
        else:
            groundtrue=img.clone()
            
        if self.Randomnoise is True :
            img += torch.randn(img.size())*self.sigma*torch.rand(1)
                
        else:
            img += torch.randn(img.size())*self.sigma
                

        return img, groundtrue


    def __len__(self):
        return len(self.imgs)










