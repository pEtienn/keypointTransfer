# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:01:27 2020

@author: Etenne Pepyn
"""
import numpy as np 
from numba import guvectorize,float64,int64



@guvectorize([(int64[:,:,:],int64[:,:,:],int64[:,:,:])], '(x,y,z),(a,b,c)->(a,b,c)',nopython=True)
def ReadyFor(im,dummy,imOutPadded):
    """
     resizes 3D image using NN
     *** INPUT ***
    im: image
    dum: dummy array to have dimensions for the output array
    im2: output array
     *** OUTPUT ***
    new resized Image
    """
    [row,col,deep]=np.asarray(np.shape(im))
    [row2,col2,deep2]=np.asarray(np.shape(dummy))
    for i in range(1,row2,2):
            for j in range(1,col2,2):
                for k in range(1,deep2,2):
                    imOutPadded[i,j,k]=im[np.int64(i/2),np.int64(j/2),np.int64(k/2)]
            
    for i in range(1,int(row*2)):
        for j in range(1,int(col*2)):
            for k in range(1,int(deep*2)):
                if imOutPadded[i,j,k]==0:
                    cube=imOutPadded[i-1:i+2,j-1:j+2,k-1:k+2]
                    u=np.unique(cube)
                    if u.shape[0]==2:
                       imOutPadded[i,j,k]= u[1]
                    else:
                        imOutPadded[i,j,k]=0
                        
row=120
col=120
deep=120
a=np.arange(1,int(row*col*deep+1),dtype=np.int64).reshape((row,col,deep))
a[0:2,0:2,0:2]=2
a[2,1,1]=2

bPadded=np.zeros((int(row*2+1),int(col*2+1),int(deep*2+1)),dtype=np.int64)
ReadyFor(a,np.copy(bPadded),bPadded)