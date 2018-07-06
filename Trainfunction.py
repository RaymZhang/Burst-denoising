
#cd desktop/denoising

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Savemodel import Save_model
from Savemodel import Save_modelloss
from PSNR import psnr
from Mycudatransformation import MyRandomCrop
from Burstload import Burstfolder
from ImagesloaderRAM import *
from time import time
import numpy as np
import torchvision.utils as tu
import pylab as plt

dtype = torch.FloatTensor
gpu_dtype=torch.cuda.FloatTensor
##


"""
Fonction permettant d'entrainer un reseau de neurones non récursif pour le debruitage.
Input :
    - model : reseau à entrainer
    - loader_train : l'ensemble d'image issu du dataloader
    - optimizer : la methode de descente de gradient
    - num_epoch : le nombre d'epoch d'entrainement
    - save_every : sauvegarde le model tous les n epoch
    - loss_every : print le loss tous les n epoch
"""
def train(model,loader_train, loss_fn, optimizer, scheduler, num_epochs = 1,save_every=1,loss_every=10):
    
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        
        model.train()
        
        for t, (x, y) in enumerate(loader_train):
            if torch.cuda.is_available():
                x_var = Variable(x.type(gpu_dtype))
                y_var = Variable(y.type(gpu_dtype))
            
            else: 
                x_var = Variable(x.type(dtype))
                y_var = Variable(y.type(dtype))

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            
            if (t + 1) % loss_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data))
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step(loss.data)
        
        if (epoch+1) % save_every == 0:
            Sauve = input('Save the model ? : y for yes ')
            
            if Sauve is 'y':
                filename=input('Name of the file ?')
                Save_model(model,optimizer,filename)
            
            
  
  
"""
Fonction permettant d'entrainer un reseau de neurones non récursif pour le debruitage SUR UN SERVEUR.
Input :
    - model : reseau à entrainer
    - loader_train : l'ensemble d'image issu du dataloader
    - optimizer : la methode de descente de gradient
    - num_epoch : le nombre d'epoch d'entrainement
    - save_every : sauvegarde le model tout les n epoch
    - filename : nom de la sauvegarde filename+nombre d'epoch 
    - loss_every : print le loss touts les n epoch
"""

###

def trainserver(model,loader_train, loss_fn, optimizer,scheduler, num_epochs = 1,save_every=1,loss_every=10,filename='denoiser'):
    
    loss_history=[]
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        
        for t, (x, y) in enumerate(loader_train):
            if torch.cuda.is_available():
                
                x_var = Variable(x.type(gpu_dtype))
                y_var = Variable(y.type(gpu_dtype))
            
            else: 
                x_var = Variable(x.type(dtype))
                y_var = Variable(y.type(dtype))

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            
            if (t + 1) % loss_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data))
                loss_history.append(loss)
                

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % save_every == 0:
                Save_modelloss(model,optimizer,loss_history,filename+'%s ' %int(epoch+1))
                
    
        scheduler.step(loss.data)
        
        
        
            
   
###   Trainfunction pour burst
   


   
def trainburst(model,loader_train, loss_fn, optimizer, scheduler, Nb_frames = 8, num_epochs = 1,save_every=1,loss_every=10):
    
    
    mfinit1,mfinit2,mfinit3,mfinit4,mfinit5,mfinit6,mfinit7=torch.zeros(7,loader_train.batch_size,64,64,64).cuda()
    mfinit8=torch.zeros(loader_train.batch_size,1,64,64).cuda()
    
    model.train()

    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        
        
        
        
        for t, (x, y) in enumerate(loader_train):
            if torch.cuda.is_available():
                x_var = Variable(torch.transpose(x.type(gpu_dtype),0,1))
                y_var = Variable(torch.transpose(y.type(gpu_dtype),0,1))
            
            
            else: 
                x_var = Variable(torch.transpose(x.type(dtype),0,1))
                y_var = Variable(torch.transpose(y.type(dtype),0,1))
                
              
                        
            loss=0
            i=0
            for frame,target in zip(x_var,y_var):
                if i==0 :
                    i+=1
                    frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8 = model(frame,mfinit1,mfinit2,mfinit3,mfinit4,mfinit5,mfinit6,mfinit7,mfinit8)
                else:
                    frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8 = model(frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8)
                
                loss += loss_fn(frame,target)+loss_fn(mf8,target)
            
            if (t + 1) % loss_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data))
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step(loss.data)
        
        if (epoch+1) % save_every == 0:
            Sauve = input('Save the model ? : y for yes ')
            
            if Sauve is 'y':
                filename=input('Name of the file ?')
                Save_model(model,optimizer,filename) 
        
        
                
    
        
   


##
def trainburstserveur(model,paths, loss_fn, optimizer, scheduler,name, Nb_frames = 4,batch_size=10, num_epochs = 100,nb_subepoch=1000,save_every=1):
    
    
    tic=time()
    model.train()
    
    loss_history=[]
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        
        for path in paths:
            trainset = 0
            trainloader = 0
            trainset = Burstfolder(path,0.1,Nb_frames,MyRandomCrop(64),Randomnoise=False, loader= RGB_loader,loadram='cpu')
            trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size,drop_last=True)
            
            mfinit1,mfinit2,mfinit3,mfinit4,mfinit5,mfinit6,mfinit7=torch.zeros(7,trainloader.batch_size,64,64,64).cuda()
            mfinit8=torch.zeros(trainloader.batch_size,3,64,64).cuda()
            
            
            for subepoch in range(nb_subepoch):
                tac=time()
                if tac-tic > 36000 :
                    Save_modelloss(model,optimizer,loss_history,'MFDC'+'%s ' %int(epoch+1))
                    tic=time()
                    
                
        
                for t, (x, y) in enumerate(trainloader):
                    
                    x_var = Variable(torch.transpose(x.type(gpu_dtype),0,1))
                    y_var = Variable(torch.transpose(y.type(gpu_dtype),0,1))
                    
                    i=0
                    for frame,target in zip(x_var,y_var):
                        if i==0 :
                            i+=1
                            frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8 = model(frame,mfinit1,mfinit2,mfinit3,mfinit4,mfinit5,mfinit6,mfinit7,mfinit8)
                            loss = loss_fn(frame,target)+loss_fn(mf8,target)
                            
                        else:
                            frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8 = model(frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8)                        
                            loss += loss_fn(frame,target)+loss_fn(mf8,target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                print('subepoch = %d, loss = %.4f' % (subepoch + 1, loss.data))
                loss_history.append(loss)
                scheduler.step(loss.data)
        
        print('epoch = %d, loss = %.4f' % (epoch + 1, loss.data))        
        if (epoch+1) % save_every == 0:
                Save_modelloss(model,optimizer,loss_history,name+'%s ' %int(epoch+1))
        
        
####


"""
Fonction permettant de montrer les resultats de debuitages d'un reseau de neuronnes
Input :
    - model : reseau à tester
    - loader_train : l'ensemble d'image issu du dataloader
    - pause : temps de pause entre chaque image
    - check : if True cela affiche la moyenne de l'image, le min , le max
    
Affiche :
    -L'image normale
    -L'image bruitée
    -L'image debruité
    -Les PSNR
"""


def Show_result(Denoiser, loader, pause, check =False):
    Denoiser.eval()
     
    plt.ion()
    plt.show()
    
    
    #Calcul de la PSNR moyenne
    PSNRmoy=0
    Compteur=0
    
    for i, data in enumerate(loader):
        images, groundtrue = data
        
        plt.figure(1)
        plt.clf()
        plt.imshow(np.clip(np.transpose(tu.make_grid(groundtrue,range=(0,1), nrow=4).numpy(),(1,2,0)),0,1))
        plt.axis('off')
        
        
        print("Image bruitée")
        plt.figure(2)
        plt.clf()
        plt.imshow(np.clip(np.transpose(tu.make_grid(images,range=(0,1), nrow=4).numpy(),(1,2,0)),0,1))
        plt.axis('off')
        
        
        A=psnr(groundtrue.numpy(),images.numpy(),check=check)
        plt.title(r'%f ' %A)
        
        
        print("Image débruitée")
        plt.figure(3)
        plt.clf()
        images=Denoiser(Variable(images,requires_grad=False).type(gpu_dtype)).data.cpu()
        plt.imshow(np.clip(np.transpose(tu.make_grid(images,range=(0,1), nrow=4).numpy(),(1,2,0)),0,1))
        plt.axis('off')
        
        A=psnr(groundtrue.numpy(),images.numpy(),check=check)
        PSNRmoy+=A
        Compteur+=1
        plt.title(r'%f ' %A)
        
        plt.pause(pause)
    
    print('PSNR moyen = %f '%(PSNRmoy/Compteur))


"""
Fonction permettant de comparer les resultats de debuitages d'un reseau de neuronnes
Input :
    - Denoiser1 : debruiteur 1 à tester
    - Denoiser2 : debruiteur 2 à tester
    - loader_train : l'ensemble d'image issu du dataloader
    - pause : temps de pause entre chaque image
    - check : if True cela affiche la moyenne de l'image, le min , le max
    
Affiche :
    - L'image normale
    - L'image bruitée
    - L'image debruité par 1
    - L'image debruité par 2
    - Les PSNR
"""    
    
def Show_result2(Denoiser1,Denoiser2, loader, pause,check=False):
    Denoiser1.eval()
    Denoiser2.eval()
     
    plt.ion()
    plt.show()
    
    
    #Calcul de la PSNR moyenne
    PSNRmoy1=0
    PSNRmoy2=0
    Compteur=0
    
    for i, data in enumerate(loader):
        images, groundtrue = data
        
        plt.figure(1)
        plt.clf()
        plt.imshow(np.clip(np.transpose(tu.make_grid(groundtrue,range=(0,1), nrow=4).numpy(),(1,2,0)),0,1))
        plt.axis('off')
        
        
        print("Image bruitée")
        plt.figure(2)
        plt.clf()
        plt.imshow(np.clip(np.transpose(tu.make_grid(images,range=(0,1), nrow=4).numpy(),(1,2,0)),0,1))
        plt.axis('off')
        
        
        A=psnr(groundtrue.numpy(),images.numpy(),check=check)
        plt.title(r'%f ' %A)
        
        
        print("Image débruitée par 1")
        plt.figure(3)
        plt.clf()
        images1=Denoiser1(Variable(images,requires_grad=False).type(gpu_dtype)).data.cpu()
        plt.imshow(np.clip(np.transpose(tu.make_grid(images1,range=(0,1), nrow=4).numpy(),(1,2,0)),0,1))
        plt.axis('off')
        
        A=psnr(groundtrue.numpy(),images1.numpy(),check=check)
        PSNRmoy1+=A
        Compteur+=1
        plt.title(r' Denoiseur1 PSNR=%f ' %A)
        
        print("Image débruitée par 2")
        plt.figure(4)
        plt.clf()
        images2=Denoiser2(Variable(images,requires_grad=False).type(gpu_dtype)).data.cpu()
        plt.imshow(np.clip(np.transpose(tu.make_grid(images2,range=(0,1), nrow=4).numpy(),(1,2,0)),0,1))
        plt.axis('off')
        
        A=psnr(groundtrue.numpy(),images2.numpy())
        PSNRmoy2+=A
        plt.title(r' Denoiseur2 PSNR= %f ' %A)
        
        
        
        
        plt.pause(pause)
    
    print(' Mean PSNR denoiseur1 = %f '%(PSNRmoy1/Compteur))
    print(' Mean PSNR denoiseur2 = %f '%(PSNRmoy2/Compteur))
 
###



"""
Fonction permettant d'afficher les resultats de debuitages d'un reseau de neuronnes. Only work with RGB images.
Input :
    - Denoiser : debruiteur 1 à tester
    - loader_train : l'ensemble de paquet de burst issu du dataloader (burstload)
    - pause : temps de pause entre chaque burst
    - framerate : temps entre chaque image du burst
    - check : if True cela affiche la moyenne de l'image, le min , le max ( A ne plus utiler ici )
    
Affiche :
    - La première image normale
    - La première image bruitée
    - La première image débruitée
    - Les images normale du burst successivement
    - Les images bruitée du burst successivement
    - Les images débruité du burst successivement
    - Les PSNR :
        - Moyenne premiere image
        - Moyenne dernière image
        - De chaque image 
"""    

def Show_burst(Denoiser,loader,pause, framerate =0.1, check=False):
    Denoiser.eval()
    
    with torch.no_grad():
        
        
        plt.ion()
        plt.show()
        
        #Calcul de la PSNR moyenne
        PSNRmoyLast=0
        PSNRmoyFirst=0
        PSNRmoySFDLast=0
        PSNRmoySFDFirst=0
        
        Compteur=0
        
        for t, (x,y) in enumerate(loader):
            
            x_var = Variable(torch.transpose(x.type(gpu_dtype),0,1),requires_grad=False)
            y = torch.transpose(y,0,1)
            i=0
            
            for frame,target in zip(x_var,y):
                if i==0 :
                
                    images=target[0]
                    plt.figure(1)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    
                    c,h,w=images.shape
                    
                    images=frame.data[0].cpu()
                    print("Image bruitée 1")
                    plt.figure(2)
                    plt.clf()
                    
                    images=np.clip(images.numpy(),0,1)
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    mfinit1,mfinit2,mfinit3,mfinit4,mfinit5,mfinit6,mfinit7=torch.zeros(7,loader.batch_size,64,h,w,requires_grad=False).cuda()
                    mfinit8=torch.zeros(loader.batch_size,c,h,w,requires_grad=False).cuda() 
                    
                    
                    i+=1
                    frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8 = Denoiser(frame,mfinit1,mfinit2,mfinit3,mfinit4,mfinit5,mfinit6,mfinit7,mfinit8)
                    
                    images=frame.data[0].cpu()
                    print("Image 1 debruité SFD")
                    plt.figure(3)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    PSNRmoySFDFirst+=A
                    
                    
                    images=mf8.data[0].cpu()
                    print("Image 1 debruité MFD")
                    plt.figure(4)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    PSNRmoyFirst+=A
                    
                    torch.cuda.empty_cache()
                    
                else:
                    i+=1
                    images=target[0]
                    plt.figure(5)
                    plt.clf()
                    images=np.clip(images,0,1)
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    
                    images=frame.data[0].cpu()
                    print("Image %d bruitée" %i)
                    plt.figure(6)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    
                    frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8 =Denoiser(frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8)
                    
                    images=frame.data[0].cpu()
                    print("Image %d debruitée SFD" %i)
                    plt.figure(7)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    B=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %B)
                    
                    images=mf8.data[0].cpu()
                    print("Image %d debruitée MFD " %i)
                    plt.figure(8)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    torch.cuda.empty_cache()
                    plt.pause(framerate)
                    
            PSNRmoySFDLast+=B
            PSNRmoyLast+=A
            Compteur+=1
            plt.pause(pause)
        
        print(' Mean PSNR MFD First = %f '%(PSNRmoyFirst/Compteur))
        print(' Mean PSNR MFD Last = %f '%(PSNRmoyLast/Compteur))
        print(' Mean PSNR SFD Last = %f '%(PSNRmoySFDLast/Compteur))
        print(' Mean PSNR SFD First = %f '%(PSNRmoySFDFirst/Compteur))


##


def Show_burst2(Denoiser,SFD,loader,pause, framerate =0.1, check=False):
    Denoiser.eval()
    SFD.eval()
    
    with torch.no_grad():
        
        
        plt.ion()
        plt.show()
        
        #Calcul de la PSNR moyenne
        PSNRmoyLast=0
        PSNRmoyFirst=0
        PSNRmoyMSFDLast=0
        PSNRmoyMSFDFirst=0
        PSNRmoySFDLast=0
        PSNRmoySFDFirst=0
        
        Compteur=0
        
        for t, (x,y) in enumerate(loader):
            
            x_var = Variable(torch.transpose(x.type(gpu_dtype),0,1),requires_grad=False)
            y = torch.transpose(y,0,1)
            i=0
            
            for frame,target in zip(x_var,y):
                if i==0 :
                
                    images=target[0]
                    plt.figure(1)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    
                    c,h,w=images.shape
                    
                    images=frame.data[0].cpu()
                    print("Image bruitée 1")
                    plt.figure(2)
                    plt.clf()
                    
                    images=np.clip(images.numpy(),0,1)
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    mfinit1,mfinit2,mfinit3,mfinit4,mfinit5,mfinit6,mfinit7=torch.zeros(7,loader.batch_size,64,h,w,requires_grad=False).cuda()
                    mfinit8=torch.zeros(loader.batch_size,c,h,w,requires_grad=False).cuda() 
                    
                    
                    i+=1
                    frame1,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8 = Denoiser(frame,mfinit1,mfinit2,mfinit3,mfinit4,mfinit5,mfinit6,mfinit7,mfinit8)
                    
                    
                    frame=SFD(frame)
                    
                    images=frame1.data[0].cpu()
                    print("Image 1 debruité MSFD")
                    plt.figure(3)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    PSNRmoyMSFDFirst+=A
                    
                    images=frame.data[0].cpu()
                    print("Image 1 debruité MSFD")
                    plt.figure(4)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    PSNRmoySFDFirst+=A
                    
                    
                    images=mf8.data[0].cpu()
                    print("Image 1 debruité MFD")
                    plt.figure(5)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    PSNRmoyFirst+=A
                    
                    torch.cuda.empty_cache()
                    
                else:
                    i+=1
                    images=target[0]
                    plt.figure(6)
                    plt.clf()
                    images=np.clip(images,0,1)
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    
                    images=frame.data[0].cpu()
                    print("Image %d bruitée" %i)
                    plt.figure(7)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    
                    frame1,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8 =Denoiser(frame,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8)
                    
                    frame=SFD(frame)
                    
                    images=frame1.data[0].cpu()
                    print("Image %d debruitée MSFD" %i)
                    plt.figure(8)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    B=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %B)
                    
                    
                    images=frame.data[0].cpu()
                    print("Image %d debruité SFD" %i)
                    plt.figure(9)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    D=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %D)
                    
                                        
                    images=mf8.data[0].cpu()
                    print("Image %d debruitée MFD " %i)
                    plt.figure(10)
                    plt.clf()
                    images=np.clip(images.numpy(),0,1)
                    
                    plt.imshow(np.transpose(images,(1,2,0)))
                    plt.axis('off')
                    A=psnr(target[0].numpy(),images,check=check)
                    plt.title(r'%f ' %A)
                    
                    torch.cuda.empty_cache()
                    plt.pause(framerate)
                    
            PSNRmoyMSFDLast+=B
            PSNRmoyLast+=A
            PSNRmoySFDLast+=D
            Compteur+=1
            plt.pause(pause)
        
        print(' Mean PSNR MFD First = %f '%(PSNRmoyFirst/Compteur))
        print(' Mean PSNR MFD Last = %f '%(PSNRmoyLast/Compteur))
        print(' Mean PSNR MSFD Last = %f '%(PSNRmoyMSFDLast/Compteur))
        print(' Mean PSNR SFD First = %f '%(PSNRmoyMSFDFirst/Compteur))
        print(' Mean PSNR SFD Last = %f '%(PSNRmoySFDLast/Compteur))
        print(' Mean PSNR SFD First = %f '%(PSNRmoySFDFirst/Compteur))






    


        
    