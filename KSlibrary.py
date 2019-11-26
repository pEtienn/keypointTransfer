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
from scipy.ndimage import gaussian_filter
from pyflann import *

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

def WriteKeyFile(path,keys,header='default'):
    #todo change the number of features on given headers
    if header=='default':
        header="# featExtract 1.1 \n# Extraction Voxel Resolution (ijk) : 176 208 176\
        \nExtraction Voxel Size (mm)  (ijk) : 1.000000 1.000000 1.000000\
        \nFeature Coordinate Space: millimeters (gto_xyz)\nFeatures:" +str(keys.shape[0])+"\
        \nScale-space location[x y z scale] orientation[o11 o12 o13 o21 o22 o23 o31 o32 o32] 2nd moment eigenvalues[e1 e2 e3] info flag[i1] descriptor[d1 .. d64]\n"
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
        WriteKeyFile(p,k2,h)

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
        for i in range(k2.shape[0]):
            if np.sum(np.all(k1-k2[i,:]<1e-005,axis=1))==1:
                s+=1
    else:
        for i in range(k1.shape[0]):
            if np.sum(np.all(k2-k1[i,:]<1e-005,axis=1))==1:
                s+=1
    return s     

def SubstractKeyImages(positive,negative):
    rest=np.copy(positive)
    for i in range(rest.shape[0]):
        if np.sum(np.all(negative-rest[i,:]<1e-005,axis=1))==1:
            rest[i,:]=0
    out=rest[~np.all(rest==0,axis=1)]
    return out

def FilterKeysWithMask(k,mask):
    k2=np.zeros(k.shape)
    for i in range(k.shape[0]):
        if mask[tuple(np.int32(k[i,kIP.XYZ]))]==True:
            k2[i,:]=k[i,:]
    k2=k2[~np.all(k2==0,axis=1)]
    k3=SubstractKeyImages(k,k2)
    if (k3.shape[0]+k2.shape[0])!=k.shape[0]:
        print('problem in FilterKeysWithMask')
    return k2



def CreateMaskKeyFiles(maskP,keyTestP,keyMaskP):
    maskF=os.listdir(maskP)
    keyTestF=os.listdir(keyTestP)
    for f in maskF:
        s1='_masked_gfc_reg.hdr'
        if s1 in f:
            n=f[:9]
            print(n)
            keyTestFP=os.path.join(keyTestP,[x for x in keyTestF if n in x][0])
            maskFP=os.path.join(maskP,f)
            keyMaskFP=os.path.join(keyMaskP,f[:-3]+'key')
            [k,r,h]=GetKeypointsResolutionHeader(keyTestFP)
            img=nib.load(maskFP)
            mask=np.squeeze(img.get_fdata())>0
            brainK=FilterKeysWithMask(k,mask)
            WriteKeyFile(keyMaskFP,brainK,header=h)
#            mat=ut.getKeypointFromOneFile(keyTestFP)
#            fW = open(keyMaskFP,"w", newline="\n")
#            fR = open(keyTestFP,"r")
#            for i in range(6):
#                fW.write(fR.readline())
#            

#            for i in range(mat.shape[0]):
#                if arr[tuple(np.int32(mat[i,kIP.XYZ]))]==True:
#                    fW.write(fR.readline())
#    
#    fW.close()
#    fR.close()

def GetSubCube(array,lowBound,shapeSubCube):
    """
    each bound is a 3x1 dimension array
    """
    a=array
    lb=lowBound
    hb=shapeSubCube
    return a[lb[0]:lb[0]+hb[0],
             lb[1]:lb[1]+hb[1],
             lb[2]:lb[2]+hb[2]]


def GenerateMask(listMatches,keyTrainingData,keyTest,resolutionTest,kIP,maskPaths,allKeyMaskPaths,patientName):
    varPerMatches=np.zeros(len(listMatches))
    dXYZ=np.zeros((len(listMatches),3))
    for i in range(len(listMatches)):
       
        matches=listMatches[i]
        keyTrainingDatum=keyTrainingData[i]
        
        testXYZ=keyTest[np.int32(matches[:,0]),kIP.XYZ]
        trainingXYZ=keyTrainingDatum[np.int32(matches[:,1]),kIP.XYZ]
        matchedXYZDifference=testXYZ-trainingXYZ      
        varPerMatches[i]=np.var(matchedXYZDifference[:,0])+np.var(matchedXYZDifference[:,1])+np.var(matchedXYZDifference[:,2]) #TO IMPROVE
        [unused,dXYZ[i,:]]=kt.houghTransformGaussian(matchedXYZDifference)
    
    
    brainMask=CreateCombinedMask(maskPaths,varPerMatches,dXYZ,resolutionTest)
    skullMask=CreateCombinedMask(maskPaths,varPerMatches,dXYZ,resolutionTest,brainMask=False) # skull and background ==1
    [keyTrueBrain,r,h]=GetKeypointsResolutionHeader([x for x in allKeyMaskPaths if patientName in x][0])
    return [keyTrueBrain,brainMask,skullMask]

def GenerateNormalizedProbabilityMap(pK,keyTest,brainShape):
    pMap=np.zeros(np.append(np.asarray(brainShape),2))
    for i in range(keyTest.shape[0]):
        scale=keyTest[i,kIP.scale]
        s=int(scale)
        size=int(int(scale)*2+1)
        mid=int((size+1)/2)
        probabilityPoint=np.zeros((size,size,size,2))
        probabilityPoint[mid,mid,mid,:]=pK[i,:]
        probabilityPoint[:,:,:,0]=gaussian_filter(probabilityPoint[:,:,:,0],sigma=scale)
        probabilityPoint[:,:,:,1]=gaussian_filter(probabilityPoint[:,:,:,1],sigma=scale)       
        XYZ=np.int64(np.asarray(keyTest[i,kIP.XYZ]))
        pMap[XYZ[0]-s:XYZ[0]+s+1,XYZ[1]-s:XYZ[1]+s+1,XYZ[2]-s:XYZ[2]+s+1,:]=probabilityPoint
    return pMap

def GetProbabilityKeyFromPMap(pMap,keys):
    pK=np.zeros((keys.shape[0],2))
    for i in range(keys.shape[0]):
        pK[i,:]=pMap[int(keys[i,0]),int(keys[i,1]),int(keys[i,2]),:]
    return pK

def ClassifyByRatio(probabilityKey,desiredRatio=0.95):
    d=1
    f=1
    pK=probabilityKey
    while d>0.1:
        mask=f*pK[:,0]>pK[:,1]
        nB=np.sum(mask)
        nS=np.sum(~mask)
        r=nB/(nB+nS)
        d=desiredRatio-r
        f=f*(1+d)
    indexBrain=f*pK[:,0]>pK[:,1]
    return indexBrain
        
def GenerateTrainingDatabase(srcKeyPath,destPath,maskPath,numberDetectionRegex='OAS._([0-9]{4})_',maskFileType='hdr'  ):
    rPatientNumber=re.compile(numberDetectionRegex)
    srcKeyPath=str(Path(srcKeyPath))
    destPath=str(Path(destPath))
    maskPath=str(Path(maskPath))
    
    allF=os.listdir(srcKeyPath)
    allMask=os.listdir(maskPath)
    brainPath=os.path.join(destPath,'brainKey')
    if not os.path.exists(brainPath):
        os.mkdir(brainPath)
   
    skullPath=os.path.join(destPath,'skullKey')
    if not os.path.exists(skullPath):
        os.mkdir(skullPath)
    
    for i in range(len(allF)):
        f=allF[i]
        patientNumber=rPatientNumber.findall(f)[0]
        [k,r,h]=GetKeypointsResolutionHeader(os.path.join(srcKeyPath,f))
        maskName=[x for x in allMask if (patientNumber in x and maskFileType in x)][0]
        mask=kt.getNiiData(os.path.join(maskPath,maskName))>0
        kBrain=FilterKeysWithMask(k,mask)
        kSkull=FilterKeysWithMask(k,~mask)
        WriteKeyFile(os.path.join(brainPath,patientNumber+'.key'),kBrain,header=h)
        WriteKeyFile(os.path.join(skullPath,patientNumber+'.key'),kSkull,header=h)
        

def UnitTestGenerateTrainingDatabase(databasePath,srcKeyPath):
    srcKeyPath=str(Path(srcKeyPath))
    databasePath=str(Path(databasePath))
    brainPath=os.path.join(databasePath,'brainKey')
    skullPath=os.path.join(databasePath,'skullKey')
    allBrain=os.listdir(brainPath)
    allKey=os.listdir(srcKeyPath)
    for i in range(len(allBrain)):
        f=allBrain[i]
        name=f[:-4]
        keyFile=[x for x in allKey if name in x][0]
        [k,r,h]=GetKeypointsResolutionHeader(os.path.join(srcKeyPath,keyFile))
        [kBrain,r,h]=GetKeypointsResolutionHeader(os.path.join(brainPath,f))
        [kSkull,r,h]=GetKeypointsResolutionHeader(os.path.join(skullPath,f))
        
        combinedKey=np.concatenate((kBrain,kSkull),axis=0)
        s=CompareKeyImages(k,combinedKey)
        if s!=combinedKey.shape[0] or s!=k.shape[0]:
            print('combined: ',combinedKey.shape[0])
            print('original: ',k.shape[0])
            print('number equal: ',s)
            m1=SubstractKeyImages(combinedKey,k)
            m2=SubstractKeyImages(k,combinedKey)
            print(m1)
            print(m2)
            print('failed')

def GenerateFilePaths(trainingSetPath):
    trainingSetPath=str(Path(trainingSetPath))
    brainPath=os.path.join(trainingSetPath,'brainKey')
    skullPath=os.path.join(trainingSetPath,'skullKey')
    brainFilePaths=os.listdir(brainPath)
    skullFilePaths=os.listdir(skullPath)
    for i in range(len(brainFilePaths)):
        brainFilePaths[i]=os.path.join(trainingSetPath,'brainKey',brainFilePaths[i])
        skullFilePaths[i]=os.path.join(trainingSetPath,'skullKey',skullFilePaths[i])
    return [brainFilePaths,skullFilePaths]

def RemoveTestPatientFromFilePaths(testNumber,brainFilePaths,skullFilePaths):
    brainFilePaths=[x for x in brainFilePaths if testNumber not in os.path.basename(x)]
    skullFilePaths=[x for x in skullFilePaths if testNumber not in os.path.basename(x)]
    return [brainFilePaths, skullFilePaths]

def CombineAllKey(brainFilePaths,skullFilePaths):
    listBrainKeys=[]
    listSkullKeys=[]
    nbKey=0
    for i in range(len(brainFilePaths)):
        [k1,r,h]=GetKeypointsResolutionHeader(brainFilePaths[i])
        listBrainKeys.append(k1)
        [k2,r,h]=GetKeypointsResolutionHeader(skullFilePaths[i])
        listSkullKeys.append(k2)
        nbKey+=k1.shape[0]+k2.shape[0]
        
    allTrainingKey=np.zeros((nbKey,83))
    indexCounter=0
    for i in range(len(listBrainKeys)):
        keyBrain=listBrainKeys[i]
        keySkull=listSkullKeys[i]
        allTrainingKey[indexCounter:indexCounter+keyBrain.shape[0],:-1]=np.concatenate((keyBrain,np.ones((keyBrain.shape[0],1))),axis=1)
        allTrainingKey[indexCounter:indexCounter+keyBrain.shape[0],-1]=i
        indexCounter+=keyBrain.shape[0]
        allTrainingKey[indexCounter:indexCounter+keySkull.shape[0],:-1]=np.concatenate((keySkull,np.zeros((keySkull.shape[0],1))),axis=1)
        allTrainingKey[indexCounter:indexCounter+keySkull.shape[0],-1]=i
        indexCounter+=keySkull.shape[0]
    return allTrainingKey

def SkullStrip(testKey,allTrainingKey,nbTrainingImages,brainShape,normalizeProbabilitySpatially=False,patientRepertory=None,keyTrueBrain=None,patientName=None,doRandomTest=False):
    viewBrain=allTrainingKey[allTrainingKey[:,81]==1,:]
    viewSkull=allTrainingKey[allTrainingKey[:,81]==0,:]
    flannB = FLANN()
    flannS= FLANN()
    paramsB = flannB.build_index(viewBrain[:,kIP.descriptor], algorithm="kdtree",trees=4);
    paramsS=flannS.build_index(viewSkull[:,kIP.descriptor], algorithm="kdtree",trees=4);
    pK=np.zeros((testKey.shape[0],2))
    nbNN=nbTrainingImages
    
    for i in range(testKey.shape[0]):
        resultB, distB = flannB.nn_index(testKey[i,kIP.descriptor],nbNN, checks=paramsB["checks"])
        distAB=np.squeeze(np.asarray(distB))
        var=np.sqrt(distAB[0])
        exp=np.exp(-distAB/(2*np.power(var,2)))
        pK[i,0]=np.sum(exp)
        resultS, distS = flannS.nn_index(testKey[i,kIP.descriptor],nbNN, checks=paramsS["checks"])
        distAS=np.squeeze(np.asarray(distS))
        pK[i,1]=np.sum(np.exp(-distAS/(2*np.power(var,2))))
        
    if normalizeProbabilitySpatially==True and doRandomTest==False:
        pMap=GenerateNormalizedProbabilityMap(pK,testKey,brainShape)
        pK=GetProbabilityKeyFromPMap(pMap,testKey)
        
        mask=pMap[:,:,:,0]>pMap[:,:,:,1]
        
        keyBrain=FilterKeysWithMask(testKey,mask)
        keySkull=FilterKeysWithMask(testKey,~mask)
    elif doRandomTest==False:
           indexBrain=pK[:,0]>pK[:,1]
           keyBrain=testKey[indexBrain]
           keySkull=testKey[~indexBrain]
    else:
        r=np.random.rand((pK.shape[0]))
        indexBrain=(r<=0.90)
        keyBrain=testKey[indexBrain]
        keySkull=testKey[~indexBrain]

    if patientRepertory!=None:
        patientObject=Patient(testKey,keyTrueBrain,keyBrain=keyBrain,keySkull=keySkull)
        patientRepertory.AddPatient(patientName,patientObject)
        patientObject.PrintStats()
    return [keyBrain,keySkull]
    
        
class Patient:
    
    def __init__(self, keyTest,keyTrueBrain,maskBrain=None,maskSkull=None,maskTrueBrain=None,keyBrain=None,keySkull=None):
        if np.any(maskBrain) and np.any(maskSkull) and np.any(maskTrueBrain):
            self.hasMask=1
            self.kTest=keyTest
            self.ktBrain=keyTrueBrain
            self.mBrain=maskBrain 
            self.mSkull=maskSkull
            self.ktSkull=SubstractKeyImages(self.kTest,self.ktBrain)
            self.mtBrain=maskTrueBrain
             
            self.mMargin=~(self.mBrain+self.mSkull)
            self.kBrain=FilterKeysWithMask(self.kTest,self.mBrain)
            self.kMargin=FilterKeysWithMask(self.kTest,self.mMargin)
            self.kSkull=FilterKeysWithMask(self.kTest,self.mSkull)
            
            self.tp=CompareKeyImages(self.kBrain,self.ktBrain)
            self.fp=CompareKeyImages(self.kBrain,self.ktSkull)
            self.tn=CompareKeyImages(self.kSkull,self.ktSkull)
            self.fn=CompareKeyImages(self.kSkull,self.ktBrain)
            self.marginBrain=CompareKeyImages(self.kMargin,self.ktBrain)
            self.marginSkull=CompareKeyImages(self.kMargin,self.ktSkull)
            self.mDC=ut.getDC(self.mBrain,self.mtBrain,1)
            self.kDC=2*self.tp/(self.kBrain.shape[0]+self.ktBrain.shape[0])
        elif np.any(keyBrain) and np.any(keySkull):
            self.hasMask=0
            self.kTest=keyTest
            self.ktBrain=keyTrueBrain
            self.ktSkull=SubstractKeyImages(self.kTest,self.ktBrain)
            self.kBrain=keyBrain
            self.kSkull=keySkull
            self.tp=CompareKeyImages(self.kBrain,self.ktBrain)
            self.fp=CompareKeyImages(self.kBrain,self.ktSkull)
            self.tn=CompareKeyImages(self.kSkull,self.ktSkull)
            self.fn=CompareKeyImages(self.kSkull,self.ktBrain)
            self.kBrainDC=2*self.tp/(self.kBrain.shape[0]+self.ktBrain.shape[0])
            self.kSkullDC=2*self.tn/(self.kSkull.shape[0]+self.ktSkull.shape[0])
        
        
    def PrintStats(self):
        if self.hasMask==1:
            print('brain: ',self.kBrain.shape[0])
            print('margin: ',self.kMargin.shape[0])
            print('skull: ',self.kSkull.shape[0])
            print('|true Brain|: ',self.ktBrain.shape[0])
            print('|true positive|: ',self.tp)
            print('|false positive|: ',self.fp)
            print('|true negative|: ',self.tn)
            print('|false negative|: ' ,self.fn)
            print('brain in margin: ',self.marginBrain)
            print('skull in margin: ',self.marginSkull)
            print('mask dc: ',self.mDC)
            print('key brain dc: ',self.kBrainDC)
            print('key skull dc: ',self.kSkullDC)
            print('\n')
        else:
            print('classified brain: ',self.kBrain.shape[0])
            print('classified skull: ',self.kSkull.shape[0])
            print('|true Brain|: ',self.ktBrain.shape[0])
            print('|true positive|: ',self.tp)
            print('|false positive|: ',self.fp)
            print('|true negative|: ',self.tn)
            print('|false negative|: ' ,self.fn)
            print('key brain dc: ',self.kBrainDC)
            print('key skull dc: ',self.kSkullDC)
            print('\n')
            
class PatientRepertory:
    
    def __init__(self):
        self.__patientDict={}
        self.mDC=[]
        self.allKBrainDC=[]
        self.allKSkullDC=[]
    
    def AddPatient(self,patientName,patientObject):
        self.__patientDict[patientName]=patientObject
        self.allKBrainDC.append(patientObject.kBrainDC)
        self.allKSkullDC.append(patientObject.kSkullDC)
    
    def GetAvgKeyBrainDC(self):
        return np.mean(np.asarray(self.allKBrainDC))
    
    def GetAvgKeySkullDC(self):
        return np.mean(np.asarray(self.allKSkullDC))        
    
    def GetAvgMaskDC(self):
        return np.mean(np.asarray(self.mDC))
    
    def GetPatient(self,patientName):
        return self.__patientDict[patientName]
    
    def ListPatient(self):
                return self.__patientDict.keys()
        
       