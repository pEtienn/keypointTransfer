# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:16:29 2020

@author: Etenne Pepyn
"""


import random
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import KSlibrary as ks
import os
import shutil
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.neighbors import KernelDensity
import time
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndi
import KSlibrary




        
poK=r'S:\HCP_NoSkullStrip_T1w\originalKeysVoxel'
pM=r'S:\HCP_NoSkullStrip_T1w\ss_images'
pssK=r'S:\HCP_NoSkullStrip_T1w\ss_images\key'
pNK=r"S:\HCP_NoSkullStrip_T1w\originalKeysVoxel_test\new"
l=[pssK,poK]
for a in l:
    for i in range(1,3):
        ks.FilterKeyWithShiftedMask(a,pM,distanceRatio=i,dstFolder=None)
# d1=r"key_d=1"
# d2=r"key_d=2"
# distanceRatio=1
# m=9
# scales=np.zeros(m)
# for i in range(m):
#     scales[i]=1.6*np.power(2,(i/3))
    
# pK=poK
# maskF=os.listdir(pM)
# keyF=os.listdir(pK)
# for f in keyF:
#     if f[-3:]=='key':
        
#         n=f[:-4]
#         print(n)
#         keyFP=os.path.join(pK,f)
#         maskFP=os.path.join(pM,[x for x in maskF if n in x][0])

#         k=ks.ReadKeypoints(keyFP)
        
#         #GenerateMasks
#         [arr,h]=ks.ReadImage(maskFP)
#         mask=arr>0
#         lMask=[]
#         for i in  range(m):         
#             r=int(np.round(distanceRatio*scales[i]))
#             d=1+2*r
#             sphereElement=np.zeros((d,d,d))
#             drawSphere(sphereElement,r,r,r,r)
#             lMask.append(ndi.binary_erosion(mask,structure=sphereElement))
#         k2=ks.FilterKeysWithMask(k,mask,lMask,scales)
#         ks.WriteKeyFile(os.path.join(pK,d1,n+'.key'),k2)
