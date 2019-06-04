import numpy as np
import os
import matplotlib.pyplot as plt



def getStats(a):
    print("Mean:",np.mean(a))
    print("Median:",np.median(a))
    print("Var:",np.var(a))
    print("Max:",np.max(a))
    
    if np.sum(a!=0)>0:
        print("\nNon zero")
        aa=a!=0
        print("Mean:",np.mean(a[aa]))
        print("Median:",np.median(a[aa]))
        print("Var:",np.var(a[aa]))
        print("Max:",np.max(a[aa]))
        print("Min:",np.min(a[aa]))
        print("Nb:",np.sum(aa))
        print("Total:",np.sum(a))
    
def getDC(computed,truth,value):
    mapC=computed==value
    mapT=truth==value
    num=2*np.sum(np.logical_and(mapC,mapT))
    den=np.sum(mapC)+np.sum(mapT)
    return num/den

def getDataFromOneFile(filePath):
    
    file= open (filePath,'r')
    #skip
    for i in range(6):
        temp=file.readline()    
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
    file.close
    return fileData

def generateAllSlices(imageTruth,imageGenerated,imageName,basePath="S:/siftTransfer/",ignoreLabelsNotInGenerated=0):
    fullPath=basePath+imageName+'/'
    os.mkdir(fullPath)
    viewNames=['sagittal','axial','coronal']
    
    uT=np.unique(imageTruth).astype(int)
    uG=np.unique(imageGenerated)
    
    if ignoreLabelsNotInGenerated==1 and uG.shape[0]>20:
        for i in range(uT.shape[0]):
            if np.sum(uG==uT[i])==0:
                imageTruth[imageTruth==uT[i]]=0

    allUniques=np.unique(np.concatenate((uT,uG)))
    sU=allUniques.shape[0]
    for i in range(sU):
        imageTruth[imageTruth==allUniques[i]]=i
        imageGenerated[imageGenerated==allUniques[i]]=i
    
    if sU<20:
        cmapName='tab20'
    else:
        cmapName='viridis'
    
    X=imageTruth.shape[0]
    Y=imageTruth.shape[0]
    Z=imageTruth.shape[0]
    XYZ=[X,Y,Z]
    canvasSagittal=np.zeros((Y,Z*2))
    canvasAxial=np.zeros((X,Z*2))
    canvasCoronal=np.zeros((X,Y*2))

    
    for j in range(3):
        folder=fullPath+viewNames[j]
        os.mkdir(folder)
        for i in range(XYZ[j]):
            
            if j==0:
                canvasSagittal[0:Y,0:Z]=imageTruth[i,:,:]
                canvasSagittal[0:Y,Z:Z*2]=imageGenerated[i,:,:]
                plt.imsave(folder+'/'+(str(i)),canvasSagittal,vmin=0,vmax=sU,cmap=cmapName)
            elif j==1:
                canvasAxial[0:X,0:Z]=imageTruth[:,i,:]
                canvasAxial[0:X,Z:Z*2]=imageGenerated[:,i,:]
                plt.imsave(folder+'/'+(str(i)),canvasAxial,vmin=0,vmax=sU,cmap=cmapName)
            else:
                canvasCoronal[0:X,0:Y]=imageTruth[:,:,i]
                canvasCoronal[0:X,Y:Y*2]=imageGenerated[:,:,i]
                plt.imsave(folder+'/'+(str(i)),canvasCoronal,vmin=0,vmax=sU,cmap=cmapName)

def getValuesInIm(im):
    u=np.unique(im)
    out=np.zeros((u.shape[0],2))
    out[:,0]=u
    for i in range(u.shape[0]):
        out[out[:,0]==u[i],1]=np.sum(im==u[i])
    return out

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def getListFileKey(commonKeyPath):
    allKeyFiles=listdir_fullpath(commonKeyPath+'keypoint')
    return allKeyFiles

def getAsegPaths(commonKeyPath):
    allNiiPaths=listdir_fullpath(commonKeyPath+'segmentation')
    return allNiiPaths

def getBrainPath(commonKeyPath):
    allBrainPaths=listdir_fullpath(commonKeyPath+'mri')
    return allBrainPaths


