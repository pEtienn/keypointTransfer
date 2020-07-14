# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:01:52 2020

@author: Etenne Pepyn
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:57:47 2020

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

import keypointTransfer as kT
import KSlibrary as KS

d=1000
w=0
lRatio=[]
iOffset=1
ratio=np.zeros((0,2))
for a in range(0,2):
    if a==0:
        iOffset=0
    elif a==1:
        iOffset=1

    for i in range(4,10):
        
        sigma=i#1.6*np.power(2,i/3)
        r=int(sigma)
        hd=int(d/2)
        I1=np.zeros((d,d))
        # I2=np.zeros((d,d,d))
        
        kT.drawCircle(I1,hd,hd,r)
        kT.fill(I1, hd)
        s1=np.sum(I1)
        
        I2=gaussian_filter(I1,sigma+iOffset)
        s2=np.sum(I2[I1>0])
        print("sigma: ",sigma,"\ns1: ",s1,"\ns2: ",s2,"\n2/1: ",s2/s1)
        ratio=np.append(ratio,[[sigma,s2/s1]],axis=0)
    lRatio.append(ratio)

