import numpy as np
import os
import matplotlib.pyplot as plt

def getStats(a):
    print("Mean:",np.mean(a))
    print("Median:",np.median(a))
    print("Var:",np.var(a))
    print("Max:",np.max(a))
    print("Non zero")
    aa=a>0
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

def generateAllSlices(image,imageName,basePath="S:/siftTransfer/"):
    fullPath=basePath+imageName+'/'
    os.mkdir(fullPath)
    viewNames=['sagittal','axial','coronal']
    for j in range(3):
        folder=fullPath+viewNames[j]
        os.mkdir(folder)
        for i in range(256):
            if j==0:
                plt.imsave(folder+'/'+(str(i)),image[i,:,:],cmap='hot')
            elif j==1:
                plt.imsave(folder+'/'+(str(i)),image[:,i,:],cmap='hot')
            else:
                plt.imsave(folder+'/'+(str(i)),image[:,:,i],cmap='hot')

def getValuesInIm(im):
    u=np.unique(im)
    out=np.zeros((u.shape[0],2))
    out[:,0]=u
    for i in range(u.shape[0]):
        out[out[:,0]==u[i],1]=np.sum(im==u[i])
    return out

def getListFileKey(commonKeyPath):
    allFolders=os.listdir(commonKeyPath)
    allKeyFiles=[]
    for file in allFolders:
        if os.path.isdir(commonKeyPath+file):
            proposedPath=commonKeyPath+file+'/'+'mri/brain.key'
            if os.path.isfile(proposedPath):
                allKeyFiles.append(proposedPath)
    return allKeyFiles

def getAsegPaths(allKeyFiles):
    allNiiPaths=[]
    for i in range(len(allKeyFiles)):
        s=allKeyFiles[i].replace('preprocessed','ABIDE_aseg')
        s=s.replace('brain.key','aseg.nii')
        if os.path.isfile(s):
            allNiiPaths.append(s)
        else:
            print("missing file:", s)
    return allNiiPaths

def getBrainPath(allKeyFiles):
    allBrainPaths=[]
    for i in range(len(allKeyFiles)):
        s=allKeyFiles[i]
        s=s.replace('brain.key','brain.nii')
        if os.path.isfile(s):
            allBrainPaths.append(s)
        else:
            print("missing file:", s)
    return allBrainPaths


