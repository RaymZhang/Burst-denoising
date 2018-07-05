import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import torchvision.transforms as T
import numpy as np

dtype = torch.FloatTensor
gpu_dtype=torch.cuda.FloatTensor
gpu_long=torch.cuda.LongTensor

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
        


## Function to make a dataset


'''
input : a folder path
return : a list containing the path of all the images in the folder
'''
def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images
  
'''
input : a folder path
return : a list containing all the images in the folder
'''
def make_dataset_ram(dir,loader=RGB_loader):
    path=make_dataset(dir)
    images=[]
    for i in path:
        images.append(loader(i))
    
    return(images)


## A load the image in the ram of the gpu useless right now




class ImageFolderRAM(data.Dataset):
    
# images must be in a folder dir
    """
    Args:
        root (string): Root directory path.
        Data_preprocessing (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        sigma : the level of noise we are going to add
        Randomnoise : if True a random noise in [0,sigma]
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        imgs (list):(image path) 
    """

    def __init__(self, root,sigma=0, Data_preprocessing=None, Randomnoise=False, loader=RGB_loader,loadram=False):
        
        
       
        if loadram == 'cpu' :
            imgs = make_dataset_ram(root,loader)
        else:
            imgs = make_dataset(root)
        
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.loadram=loadram
        self.Randomnoise=Randomnoise
        self.sigma = sigma
        self.root = root
        self.imgs = imgs
        self.Data_preprocessing = Data_preprocessing
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue 
        """
        
        if  self.loadram == 'cpu' :
            img=self.imgs[index]
        else:
            path = self.imgs[index]
            img = self.loader(path)
        
        if self.Data_preprocessing is not None:
            
            img = self.Data_preprocessing(img)
            groundtrue=img.clone()
            
        
        if self.Randomnoise is True :
            img += torch.randn(img.size())*self.sigma*torch.rand(1)
                
        else:
            img += torch.randn(img.size())*self.sigma
                

        return img, groundtrue


    def __len__(self):
        return len(self.imgs)
        
        


