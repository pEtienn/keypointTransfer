import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from matplotlib import cm
import matplotlib as mpl
import shutil
import time

def createCrossKernel(value,size):
    k=np.zeros((size,size,size))
    c=int((size-1)/2)
    k[0:size,c,c]=value
    k[c,0:size,c]=value
    k[c,c,0:size]=value
    return k
        
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

def getKeypointFromOneFile(filePath):
    
    file= open (filePath,'r')
    #skip
    for i in range(6):
        temp=file.readline()    
        if i==5 and temp[:5]!='Scale':
            print("keypointFile format not supported")
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

def generateColorBar(height,width,nbValues):
    t=nbValues/height
    mat=np.zeros((height,width))
    for i in range(height):
        mat[height-i-1,:]=t*i
    return mat.astype(int)
    
def generateAllSlices(imageTruth,imageGenerated,folderPath,listOfKeypointCoordinate,ignoreLabelsNotInGenerated=0):
    
    viewNames=['sagittal','axial','coronal']
    
    uT=np.unique(imageTruth).astype(int)
    uG=np.unique(imageGenerated)
    
    #used to reduce number of labels in image
    if ignoreLabelsNotInGenerated==1 and uT.shape[0]>=20:
        for i in range(uT.shape[0]):
            if np.sum(uG==uT[i])==0:
                imageTruth[imageTruth==uT[i]]=0
        allUniques=uG
    else:
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
    
    #generate map of keypoints transfered
    crossSize=3
    keypointMatrix=np.zeros((XYZ))
    for i in range(listOfKeypointCoordinate.shape[0]):
        [x,y,z]=listOfKeypointCoordinate[i,:]
        x=x.astype(int)
        y=y.astype(int)
        z=z.astype(int)
        keypointMatrix[x,y,z]=1
        for i in range(-(crossSize-1),(crossSize)):
            keypointMatrix[x+i,y,z]=1
            keypointMatrix[x,y+i,z]=1
            keypointMatrix[x,y,z+i]=1

    #colormap variables
    norm=mpl.colors.Normalize(vmin=0,vmax=sU)
    cmap=cmapName
    SMI=cm.ScalarMappable(norm=norm,cmap=cmap)
    
    norm1=mpl.colors.Normalize(vmin=0,vmax=1)
    cmap1=cm.hsv
    SMR=cm.ScalarMappable(norm=norm1,cmap=cmap1)
    
    for j in range(3):
        folder=os.path.join(folderPath,viewNames[j])
        os.mkdir(folder)

        for i in range(XYZ[j]):
            
            if j==0:
                width=10
                colorBar=generateColorBar(Y,width,sU)
                canvasSagittal[0:Y,0:Z]=imageTruth[i,:,:]
                canvasSagittal[0:Y,Z:Z*2]=imageGenerated[i,:,:]
                canvasSagittal[0:Y,(Z*2-width):Z*2]=colorBar
#                plt.imsave(os.path.join(folder,(str(i))),canvasSagittal,vmin=0,vmax=sU,cmap=cmapName)
                tempIm=np.uint8(SMI.to_rgba(canvasSagittal)*255)
                im = Image.fromarray(tempIm)
                imToPaste=np.uint8(SMR.to_rgba(keypointMatrix[i,:,:])*255)
                mask = Image.fromarray(np.uint8(255*(keypointMatrix[i,:,:]>0)))
                im.paste(Image.fromarray(imToPaste),(0,0),mask)
                sPath=os.path.join(folder,(str(i))+'.png')

                im.save(sPath)
#            elif j==1:
#                canvasAxial[0:X,0:Z]=imageTruth[:,i,:]
#                canvasAxial[0:X,Z:Z*2]=imageGenerated[:,i,:]
#                plt.imsave(os.path.join(folder,(str(i))),canvasAxial,vmin=0,vmax=sU,cmap=cmapName)
#            else:
#                canvasCoronal[0:X,0:Y]=imageTruth[:,:,i]
#                canvasCoronal[0:X,Y:Y*2]=imageGenerated[:,:,i]
#                plt.imsave(os.path.join(folder,(str(i))),canvasCoronal,vmin=0,vmax=sU,cmap=cmapName)


    
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
    allKeyFiles=listdir_fullpath(os.path.join(commonKeyPath,'keypoint'))
    withoutGitKeep=[x for x in allKeyFiles if '.gitkeep' not in x] #called a list comprehension
    return withoutGitKeep

def getAsegPaths(commonKeyPath):
    allNiiPaths=listdir_fullpath(os.path.join(commonKeyPath,'segmentation'))
    withoutGitKeep=[x for x in allNiiPaths if '.gitkeep' not in x] #called a list comprehension
    return withoutGitKeep

def getBrainPath(commonKeyPath):
    allBrainPaths=listdir_fullpath(os.path.join(commonKeyPath,'mri'))
    withoutGitKeep=[x for x in allBrainPaths if '.gitkeep' not in x] #called a list comprehension
    return withoutGitKeep


        


