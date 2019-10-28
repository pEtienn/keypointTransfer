# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:16:02 2019

@author: Etenne Pepyn
"""
import os
from pathlib import Path
import numpy as np 
import nibabel as nib
import re
import utilities as ut
import keypointTransfer as kt

kIP=kt.keyInformationPosition()

def GetKeypointsResolutionHeader(filePath):
    rReso=re.compile('(\d+ \d+ \d+)')
    
    file= open (filePath,'r')
    #skip
    header=''
    for i in range(6):
        r=file.readline()
        header=header+r
        if i==1:
            resoString=rReso.findall(r)[0]
            resolution=[int(i) for i in resoString.split()]
    if r[:5]!='Scale':
        print('ERROR: keypoint format not supported')
    end=0
    
    lineString=file.readline()
    fileData=np.fromstring(lineString, dtype=float,sep='\t')
    
    while end==0:
        lineString=file.readline()
        if lineString!="":
            floatLine=np.fromstring(lineString, dtype=float,sep='\t')
            fileData=np.vstack((fileData,floatLine))
        else:
            end=1
    file.close()
    return [fileData,resolution,header]

def WrittingKeyFile(path,keys,header='default'):
    if header=='default':
        header="# featExtract 1.1 \n# Extraction Voxel Resolution (ijk) : 176 208 176\
        \nExtraction Voxel Size (mm)  (ijk) : 1.000000 1.000000 1.000000\
        \nFeature Coordinate Space: millimeters (gto_xyz)\nFeatures: 4779\
        \nScale-space location[x y z scale] orientation[o11 o12 o13 o21 o22 o23 o31 o32 o32] 2nd moment eigenvalues[e1 e2 e3] info flag[i1] descriptor[d1 .. d64]"
    fW=open(path,'w',newline='\n')
    fW.write(header)
    for i in range(keys.shape[0]):
        for j in range(keys.shape[1]):
            n=str(keys[i,j])
            fW.write(n.ljust(9,'0')+'\t')
        fW.write('\n')
    fW.close()
        
def  DeleteDuplicateKey(folderPath):
    fp=str(Path(folderPath))
    allF=os.listdir(fp)
    
    for f in allF:
        p=os.path.join(fp,f)
        [k,r,h]=GetKeypointsResolutionHeader(p)
        k2=np.zeros(k.shape)
        prevXYZ=np.array([0,0,0])
        for i in range(k2.shape[0]):
            if np.all(prevXYZ!=k[i,0:3]):
                k2[i-1,:]=k[i-1,:]
            prevXYZ=k[i,0:3]
        k2=k2[~np.all(k2==0,axis=1)]
        WrittingKeyFile(p,k2,h)

def CreateCombinedMask(maskPaths,varPerMatches,dXYZ,resolutionTest,brainMask=True,maskToKeep=5):
    #*****************MASKING
    order=np.argsort(varPerMatches)
    dXYZ=np.int32(dXYZ)
    
    firstMask=(brainMask==(kt.getNiiData(maskPaths[order[0]])>0))
    marginXYZ=np.int32(2*np.amax(np.abs(dXYZ),axis=0))
    shapeMask=np.array([resolutionTest[0]+2*marginXYZ[0],resolutionTest[1]+2*marginXYZ[1],resolutionTest[2]+2*marginXYZ[2]])
    maskCombined=np.zeros(shapeMask)
    tempMask=np.zeros(shapeMask)
    t=GetSubCube(maskCombined,marginXYZ+dXYZ[order[0]],firstMask.shape)
    t+=firstMask
    
    for i in range(1,maskToKeep):
        tempMask=np.zeros(shapeMask)
        t=GetSubCube(tempMask,marginXYZ+dXYZ[order[i]],firstMask.shape)
        t+=(brainMask==(kt.getNiiData(maskPaths[order[i]])>0))
        maskCombined=np.logical_and(maskCombined,tempMask)
        
    finalMask=GetSubCube(maskCombined,marginXYZ,firstMask.shape)
    return finalMask

def CompareKeyImages(k1,k2):
    s=0
    if k1.shape[0]>k2.shape[0]:
        for i in range(k1.shape[0]):
            if np.sum(np.all(k2==k1[i,:],axis=1))==1:
                s+=1
    else:
        for i in range(k2.shape[0]):
            if np.sum(np.all(k1==k2[i,:],axis=1))==1:
                s+=1
    return s     

def FilterKeysWithMask(k,mask):
    k2=np.zeros(k.shape)
    for i in range(k.shape[0]):
        if mask[tuple(np.int32(k[i,kIP.XYZ]))]==True:
            k2[i,:]=k[i,:]
    k2=k2[~np.all(k2==0,axis=1)]
    return k2

def CreateMaskKeyFiles(maskP,keyTestP,keyMaskP):
    maskF=os.listdir(maskP)
    keyTestF=os.listdir(keyTestP)
    for f in maskF:
        s1='_masked_gfc_reg.hdr'
        if s1 in f:
            n=f[:9]
            print(n)
            keyTestFP=os.path.join(keyTestP,next(x for x in keyTestF if n in x))
            maskFP=os.path.join(maskP,f)
            keyMaskFP=os.path.join(keyMaskP,f[:-3]+'key')
            mat=ut.getKeypointFromOneFile(keyTestFP)
            fW = open(keyMaskFP,"w", newline="\n")
            fR = open(keyTestFP,"r")
            for i in range(6):
                fW.write(fR.readline())
            
            img=nib.load(maskFP)
            arr=np.squeeze(img.get_fdata())>0
            for i in range(mat.shape[0]):
                if arr[tuple(np.int32(mat[i,kIP.XYZ]))]==True:
                    fW.write(fR.readline())
    
    fW.close()
    fR.close()

def GetSubCube(a,lowBound,shapeSubCube):
    """
    each bound is a 3x1 dimension array
    """
    lb=lowBound
    hb=shapeSubCube
    return a[lb[0]:lb[0]+hb[0],
             lb[1]:lb[1]+hb[1],
             lb[2]:lb[2]+hb[2]]