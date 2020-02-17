# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:16:02 2019
import sys
sys.argv=['',r'S:\testMatch\OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc\orginalNotMatchedWithSS.key']
dir
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
import pandas

class keyInformationPosition():
    """
    information locations in SIFT datapoints
    in an array of SIFT keypoints, row are the number of the keypoints, and colums
    contains it's characteristics. Use kIP object to access the right characteristics
    """
    scale=3
    XYZ=slice(0,3,1)
    descriptor=slice(17,81,1)
    flag=16
    
kIP=keyInformationPosition()

def ReadImage(filePath):
    filePath=str(Path(filePath))
    img=nib.load(filePath)
    arr=np.squeeze(img.get_fdata())
    h=img.header
    return[arr,h]    
    
def SaveImage(filePath,arr,header):
    im2=nib.Nifti1Image(arr,affine=None,header=header)
    nib.save(im2,filePath)
    
def ReadKeypoints(filePath):
    filePath=str(Path(filePath))
    file= open (filePath,'r')
    i=0
    end=0
    while end==0:
        r=file.readline()
        if 'Scale-space' in r:
            end=1
        i=i+1
    file.close()
    a=np.arange(0,i)
    listC=list(a)
    df=pandas.read_csv(filePath,sep='\t',header=None,index_col=False,skiprows=listC)
    if len(df.columns)==82: #happens when there are tab at the end of a line (shouldnt be there)
        df=df.drop(81,axis=1)

    key=df.to_numpy(dtype=np.float64)

    return key
    
def CombineDescriptorsOfKeyFiles(srcPath,dstPath):
    allKeys=os.listdir(srcPath)
    allDes=np.empty((0,64),dtype=np.int8)
    for keyFile in allKeys:
        keys=ReadKeypoints(os.path.join(srcPath,keyFile))
        des=np.int8(keys[:,kIP.descriptor])
        allDes=np.append(allDes,des,axis=0)
        
    pandas.DataFrame(allDes).to_csv(dstPath+'.des', header=None, index=None,sep='\t')
    
def ReadDescriptors(srcPath):
    df=pandas.read_csv(srcPath,sep='\t',header=None)
    descriptors=df.to_numpy(dtype=np.int8)
    return descriptors

def GetResolutionHeaderFromKeyFile(filePath):
    rReso=re.compile('(\d+ \d+ \d+)')
    
    file= open (filePath,'r')
    #skip
    header=''
    end=0
    while end==0:
        r=file.readline()
        header=header+r
        if 'resolution' in r or 'Resolution' in r:
            resoString=rReso.findall(r)
            if resoString ==[]:
                print('resolution not found in'+filePath)
            else:
                resoString=resoString[0]
            resolution=[int(i) for i in resoString.split()]
        if 'Scale-space' in r:
            end=1
    if r[:5]!='Scale':
        print('ERROR: keypoint format not supported in'+filePath)
    file.close()
    if not('resolution' in locals()):
        resolution=np.array([0,0,0])
    return [resolution,header]

def WriteKeyFile(path,keys,header='default'):
    #todo change the number of features on given headers
    if header=='default':
        header="# featExtract 1.1\n# Extraction Voxel Resolution (ijk) : 176 208 176\
        \n#Extraction Voxel Size (mm)  (ijk) : 1.000000 1.000000 1.000000\
        \n#Feature Coordinate Space: millimeters (gto_xyz)\nFeatures: " +str(keys.shape[0])+"\
        \nScale-space location[x y z scale] orientation[o11 o12 o13 o21 o22 o23 o31 o32 o32] 2nd moment eigenvalues[e1 e2 e3] info flag[i1] descriptor[d1 .. d64]\n"
    else:
        p=re.compile('Features: \d+')
        header=p.sub('Features: '+str(keys.shape[0])+' ',header)
    fW=open(path,'w',newline='\n')
    fW.write(header)
    for i in range(keys.shape[0]):
        for j in range(keys.shape[1]):
            if j>=16:
                n=str(int(keys[i,j]))
                fW.write(n+'\t')
            else:
                n=keys[i,j]
                fW.write(format(n,'.6f')+'\t')
            
        fW.write('\n')
    fW.close()
    
def GetIndexFromMatchFile(filePath):
    filePath=str(Path(filePath))
    a=np.arange(0,4)
    listC=list(a)
    df=pandas.read_csv(filePath,sep='\t',skiprows=listC,header=None,index_col=False)
    if len(df.columns)==82: #happens when there are tab at the end of a line (shouldnt be there)
        df=df.drop(81,axis=1)
    d=df[[5]]
    size=d.shape[0]
    listIdx=[]
    for i in range(size):
        listIdx.append(int(d.iloc[i,0][-5:]))

    return listIdx
        
def  SelectKeyRotation(folderPath,rotation=True):
    fp=str(Path(folderPath))
    allF=os.listdir(fp)
    
    for f in allF:
        if f[-3:]=='key':
            p=os.path.join(fp,f)
            [r,h]=GetResolutionHeaderFromKeyFile(p)
            k=ReadKeypoints(p)
            kNoRot=np.zeros(k.shape)
            for i in range(kNoRot.shape[0]):
                if k[i,16]==0 or k[i,16]==16:
                    kNoRot[i,:]=k[i,:]
            kNoRot=kNoRot[~np.all(kNoRot==0,axis=1)]
            if rotation==False:
                WriteKeyFile(p,kNoRot,h)
            else:
                kRot=SubstractKeyImages(k, kNoRot)
                WriteKeyFile(p,kRot,h)

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

def FilterKeysWithMask(k,mask,considerScale=False):
    k2=np.zeros(k.shape)
    invMask=~mask
    prevXYZ=np.array([1,2,3])
    transfered=0
    for i in range(k.shape[0]):
        XYZ=np.int32(k[i,kIP.XYZ])
        if ~np.allclose(XYZ,prevXYZ): #used to skip keypoints with different rotation
            if considerScale==False:
                if mask[tuple(XYZ)]==True:
                    k2[i,:]=k[i,:]
                    transfered=1
                else:
                    transfered=0
            else:
                mask2=np.zeros(mask.shape,dtype=np.bool)
                c=int(k[i,kIP.scale]/np.sqrt(3)) #half the side of the smallest cube inside the keypoint
                mask2[XYZ[0]-c:XYZ[0]+c,XYZ[1]-c:XYZ[1]+c,XYZ[2]-c:XYZ[2]+c]=True
                if ~np.any(np.logical_and(invMask,mask2)):
                    k2[i,:]=k[i,:]         
                    transfered=1
                else:
                    transfered=0
        elif transfered==1:
            k2[i,:]=k[i,:]
        prevXYZ=XYZ
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
            keyTestFP=os.path.join(keyTestP,[x for x in keyTestF if n in x][0])
            maskFP=os.path.join(maskP,f)
            keyMaskFP=os.path.join(keyMaskP,f[:-3]+'key')
            k=ReadKeypoints(keyTestFP)
            [r,h]=GetResolutionHeaderFromKeyFile(keyTestFP)
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
    keyTrueBrain=ReadKeypoints([x for x in allKeyMaskPaths if patientName in x][0])
    return [keyTrueBrain,brainMask,skullMask]



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
        
def GenerateTrainingDatabase(srcKeyPath,destPath,maskPath,numberDetectionRegex='OAS._([0-9]{4})_',maskFileType='hdr' ,considerScale=False):
    rPatientNumber=re.compile(numberDetectionRegex)
    srcKeyPath=str(Path(srcKeyPath))
    destPath=str(Path(destPath))
    maskPath=str(Path(maskPath))
    
    allF=os.listdir(srcKeyPath)
    allMask=os.listdir(maskPath)
    
    for i in range(len(allF)):
        f=allF[i]
        print(f)
        patientNumber=rPatientNumber.findall(f)[0]
        path=(os.path.join(srcKeyPath,f))
        k=ReadKeypoints(path)
        [r,h]=GetResolutionHeaderFromKeyFile(path)
        maskName=[x for x in allMask if (patientNumber in x and maskFileType in x)][0]
        mask=kt.getNiiData(os.path.join(maskPath,maskName))>0
        if considerScale==False:
            kBrain=FilterKeysWithMask(k,mask)
            kSkull=FilterKeysWithMask(k,~mask)
        else:
            kBrain=FilterKeysWithMask(k,mask,considerScale=considerScale)
            kSkull=SubstractKeyImages(k,kBrain)
        kBrain[:,kIP.flag]=np.int32(kBrain[:,kIP.flag])|(1<<0)
        kSkull[:,kIP.flag]=np.int32(kSkull[:,kIP.flag])&~(1<<0)
        allKey=np.append(kBrain,kSkull,axis=0)
        WriteKeyFile(os.path.join(dstPath,patientNumber+'.key'),allKey,header=h)
        print(f)
        

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
        k=ReadKeypoints(os.path.join(srcKeyPath,keyFile))
        kBrain=ReadKeypoints(os.path.join(brainPath,f))
        kSkull=ReadKeypoints(os.path.join(skullPath,f))
        
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

def RemoveTestPatientFromFilePaths(listTestNumber,brainFilePaths,skullFilePaths):
    for testNumber in listTestNumber:
        brainFilePaths=[x for x in brainFilePaths if testNumber not in os.path.basename(x)]
        skullFilePaths=[x for x in skullFilePaths if testNumber not in os.path.basename(x)]
    return [brainFilePaths, skullFilePaths]

def CombineAllKey(brainFilePaths,skullFilePaths):
    listBrainKeys=[]
    listSkullKeys=[]
    nbKey=0
    for i in range(len(brainFilePaths)):
        k1=ReadKeypoints(brainFilePaths[i])
        listBrainKeys.append(k1)
        k2=ReadKeypoints(skullFilePaths[i])
        listSkullKeys.append(k2)
        nbKey+=k1.shape[0]+k2.shape[0]
        print('reading training file:',i)
        
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

def GenerateNormalizedProbabilityMap(pK,keyTest,brainShape):
    #to do, consider cases with smaller blank window around the brain
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
        pMap[XYZ[0]-s:XYZ[0]+s+1,XYZ[1]-s:XYZ[1]+s+1,XYZ[2]-s:XYZ[2]+s+1,:]+=probabilityPoint
        
#        maximum
#        viewpMap=pMap[XYZ[0]-s:XYZ[0]+s+1,XYZ[1]-s:XYZ[1]+s+1,XYZ[2]-s:XYZ[2]+s+1,:]
#        pMap[XYZ[0]-s:XYZ[0]+s+1,XYZ[1]-s:XYZ[1]+s+1,XYZ[2]-s:XYZ[2]+s+1,:]=np.maximum(probabilityPoint,viewpMap)
    return pMap

def GenerateSearchTree(brainDescriptor,skullDescriptor):
    flannB = FLANN()
    flannS= FLANN()
    paramsB = flannB.build_index(brainDescriptor, algorithm="kdtree",trees=8);
    paramsS=flannS.build_index(skullDescriptor, algorithm="kdtree",trees=8);
    return [[flannB,flannS],[paramsB,paramsS]]
    
def _SkullStripTest(testKey,listFlann,listParam,nbTrainingImages,brainShape,normalizeProbabilitySpatially=False,patientRepertory=None,keyTrueBrain=None,patientName=None,doRandomTest=False):
    flannB=listFlann[0]
    flannS=listFlann[1]
    paramsB=listParam[0]
    paramsS=listParam[1]
    pK=np.zeros((testKey.shape[0],2))

    for i in range(testKey.shape[0]):
        resultB, distB = flannB.nn_index(testKey[i,kIP.descriptor],nbTrainingImages, checks=paramsB["checks"])
        distAB=np.squeeze(np.asarray(distB))
        
        resultS, distS = flannS.nn_index(testKey[i,kIP.descriptor],nbTrainingImages, checks=paramsS["checks"])
        distAS=np.squeeze(np.asarray(distS))
        var=np.sqrt(np.minimum(distAB[0],distAS[0]))+1
        pK[i,0]=np.sum(np.exp(-distAB/(2*np.power(var,2))))
        pK[i,1]=np.sum(np.exp(-distAS/(2*np.power(var,2))))
#    pK=pK/testKey.shape[0]
        
    if normalizeProbabilitySpatially==True and doRandomTest==False:
        #prevent outliers from warping results too much
#        for j in range(pK.shape[1]):
#            avg=np.mean(pK[:,j])
#            var=np.var(pK[:,j])
#            pK[:,j]=np.clip(pK[:,j],avg-2*var,avg+2*var)
            
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
        print(patientName)
        patientObject.PrintStats()
    return [keyBrain,keySkull]

def _SkullStrip(testKey,listFlann,listParam,nbTrainingImages,brainShape,normalizeProbabilitySpatially=False,):
    flannB=listFlann[0]
    flannS=listFlann[1]
    paramsB=listParam[0]
    paramsS=listParam[1]
    pK=np.zeros((testKey.shape[0],2))
    testDescriptors=np.int32(testKey[:,kIP.descriptor])
    nbTrainingImages=int(nbTrainingImages)

    for i in range(testKey.shape[0]):
        resultB, distB = flannB.nn_index(testDescriptors[i,:],nbTrainingImages, checks=paramsB["checks"])
        distAB=np.squeeze(np.asarray(distB))
        
        resultS, distS = flannS.nn_index(testDescriptors[i,:],nbTrainingImages, checks=paramsS["checks"])
        distAS=np.squeeze(np.asarray(distS))
        var=np.sqrt(np.minimum(distAB[0],distAS[0]))+1
        pK[i,0]=np.sum(np.exp(-distAB/(2*np.power(var,2))))
        pK[i,1]=np.sum(np.exp(-distAS/(2*np.power(var,2))))
#    pK=pK/testKey.shape[0]
        
    if normalizeProbabilitySpatially==True:          
        pMap=GenerateNormalizedProbabilityMap(pK,testKey,brainShape)
        pK=GetProbabilityKeyFromPMap(pMap,testKey)
        
        mask=pMap[:,:,:,0]>pMap[:,:,:,1]
        
        keyBrain=FilterKeysWithMask(testKey,mask)
        keySkull=FilterKeysWithMask(testKey,~mask)
        
    else:
           indexBrain=pK[:,0]>pK[:,1]
           keyBrain=testKey[indexBrain]
           keySkull=testKey[~indexBrain]

    return [keyBrain,keySkull]

def SkullStrip(originalKeysPath,dstPath,brainTrainingSetPath=r'S:\skullStripData\trainingSet\brain.des',skullTrainingSetPath=r"S:\skullStripData\trainingSet\skull.des",printNonBrain=False,numberOfTrainingImages=None):
    allTestName=os.listdir(originalKeysPath)
    brainDescriptor=np.int32(ReadDescriptors(brainTrainingSetPath))
    skullDescriptor=np.int32(ReadDescriptors(skullTrainingSetPath))
    
    if numberOfTrainingImages==None:
        numberOfTrainingImages=brainDescriptor.shape[0]/1000
    [listFlann,listParam]=GenerateSearchTree(brainDescriptor,skullDescriptor)
    
    for i in range(len(allTestName)):
        testName=allTestName[i]
        if '.key' == testName[-4:]: 
            testPath=os.path.join(originalKeysPath,testName)        
            testKey=ReadKeypoints(testPath)
            [resolution,header]=GetResolutionHeaderFromKeyFile(testPath)    
            [keyBrain,keySkull]=_SkullStrip(testKey,listFlann,listParam,numberOfTrainingImages,resolution,normalizeProbabilitySpatially=True)
            keyBrain[:,kIP.flag]=np.int32(keyBrain[:,kIP.flag])|(1<<0)
            keySkull[:,kIP.flag]=np.int32(keySkull[:,kIP.flag])&~(1<<0)
            if printNonBrain==True:
                keyBrain=np.append(keyBrain,keySkull,axis=0)
            else:
                WriteKeyFile(os.path.join(dstPath,testName),keyBrain,header=header)

def EvaluateSkullStrip(resultPath,groundTruthPath):
    allFiles=os.listdir(resultPath)
    
    patientRepertory=PatientRepertory()
    
    for i in range(len(allFiles)):
        f=allFiles[i]
        rF=os.path.join(resultPath,f)
        gF=os.path.join(groundTruthPath,f)
        
        keyR=ReadKeypoints(rF)
        keyG=ReadKeypoints(gF)
        
        patientObject=Patient(keyR,FilterKeysByClass(keyR, 1),keyBrain=FilterKeysByClass(keyG,1),keySkull=FilterKeysByClass(keyG,0))
        patientRepertory.AddPatient(f,patientObject)
        print(f)
        patientObject.PrintStats()

def FilterKeysByClass(keys,classValue):
    return keys[np.int32(keys[:,kIP.flag])&classValue==classValue,:]
        
def GenerateCommandFileToGenerateKeyFiles(folderPath,modifiers='-i '):
    folderPath=str(Path(folderPath))
    keyFolderPath=os.path.join(folderPath,'key')
    if not os.path.exists(keyFolderPath):
        os.makedirs(keyFolderPath)
    filePath=os.path.join(folderPath,'keyCommand')
    fW=open(filePath,'w',newline='\n')
    allFiles=os.listdir(folderPath)
    for i in range(len(allFiles)):
        f=allFiles[i]
        if os.path.splitext(f)[1]=='.hdr'  or os.path.splitext(f)[1]=='.gz' or os.path.splitext(f)[1]=='.nii':
            string='./featextract.ubu '+modifiers+'./'+f+' ./key/'+f[:-4]+'.key\n'
            fW.write(string)
    fW.close()
    



def BrainExtractionOnFolder(inputFolderPath,trainingSetPath,resultPath,ignoreSameNameInTraining=False):
    inputFolderPath=str(Path(inputFolderPath))
    trainingSetPath=str(Path(trainingSetPath))
    resultPath=str(Path(resultPath))
    
    numberDetectionRegex='OAS._([0-9]{4})_'  #detect the number in path following this format:
#S:\skullStripData\keyMaskMany\OAS1_0002_MR1_mpr_nn_anon_111_t88_masked_gfc_reg.key
#use of regex will return '0002' on that path
    allKeyTestPaths=ut.listdir_fullpath(inputFolderPath)
    rPatientNumber=re.compile(numberDetectionRegex)
    listTestNumber=[]
    allTestPath=allKeyTestPaths
    for testPath in allTestPath:
        
        listTestNumber.append(os.path.basename(testPath)[:-4])
    
    [brainFilePaths, skullFilePaths]=GenerateFilePaths(trainingSetPath)
    
    if ignoreSameNameInTraining==True:
        [brainFilePaths, skullFilePaths]=RemoveTestPatientFromFilePaths(listTestNumber,brainFilePaths,skullFilePaths)
    
    allKey=CombineAllKey(brainFilePaths,skullFilePaths)
    [listFlann,listParam]=GenerateSearchTree(allKey)
    
    for i in range(len(allTestPath)):
        testPath=allTestPath[i]
        testNumber=listTestNumber[i]
        testKey=ReadKeypoints(testPath)
        [resolution,header]=GetResolutionHeaderFromKeyFile(testPath)
        resolution=np.asarray(resolution)
        mXYZ=np.mean(testKey[:,kIP.XYZ],axis=0)
        tXYZ=resolution
        dXYZ=tXYZ-mXYZ
        testKey[:,kIP.XYZ]=dXYZ+testKey[:,kIP.XYZ]
        [keyBrain,keySkull]=SkullStrip(testKey,listFlann,listParam,len(brainFilePaths),resolution*2,normalizeProbabilitySpatially=True)
        keyBrain[:,kIP.XYZ]-=dXYZ
        keySkull[:,kIP.XYZ]-=dXYZ
        writePath=os.path.join(resultPath,testNumber+'.key')
        WriteKeyFile(writePath,keyBrain,header=header)
        print(testNumber)

def ExactNN(v,matrix):
     matrix=matrix-v
     matrix=np.multiply(matrix,matrix)
     m1=np.sum(matrix,axis=1)
     result=np.argmin(m1)
     return [result,m1[result]]
    
def BrainSimilarity(brainA,brainB,paramsTraining,flannTraining):    
   flannB = FLANN()
   paramsB = flannB.build_index(brainB[:,kIP.descriptor], algorithm="kdtree",trees=8)  
   intersection=0
   for i in range(brainA.shape[0]):
        resultB, distB = flannB.nn_index(brainA[i,kIP.descriptor],1, checks=paramsB["checks"])
        resultT,distT=flannTraining.nn_index(brainA[i,kIP.descriptor],1, checks=paramsB["checks"])
#        [resultB,[distB]]=ExactNN(brainA[i,kIP.descriptor],brainB[:,kIP.descriptor])
        distB=distB[0]
        var=np.sqrt(distT[0]+1)
        intersection+=np.exp(-distB/(2*np.power(var,2)))
   similarity=intersection/(brainA.shape[0]+brainB.shape[0]-intersection)
                    
   return similarity
    
def BrainSimilarityFolder(path,trainingSetPath):
    path=str(Path(path))
    trainingSetPath=str(Path(trainingSetPath))
    [brainFilePaths, skullFilePaths]=GenerateFilePaths(trainingSetPath)
    allKey=CombineAllKey(brainFilePaths,skullFilePaths)
    flannTraining=FLANN()

    allFile=os.listdir(path)
    n=len(allFile)
    similarityMatrix=np.zeros((n,n))
    listBrain=[]
    for i in range(n):
        listBrain.append(ReadKeypoints(os.path.join(path,allFile[i])))
    allDescriptor=allKey[:,kIP.descriptor]
    for i in range(n):
        brainI=listBrain[i]
        allDescriptor=np.concatenate((allDescriptor,brainI[:,kIP.descriptor]),axis=0)
    paramsTraining=flannTraining.build_index(allKey[:,kIP.descriptor],algorithm="kdtree",trees=8)
    
    for i in range(n):
        for j in range(n):
            similarityMatrix[i,j]=BrainSimilarity(listBrain[i],listBrain[j],paramsTraining,flannTraining)
        print(i,'/',n)
    return similarityMatrix
    

class Patient:
    
    def __init__(self, keyTest,keyTestBrain,maskBrain=None,maskSkull=None,maskTrueBrain=None,keyBrain=None,keySkull=None):
        if np.any(maskBrain) and np.any(maskSkull) and np.any(maskTrueBrain):
            self.hasMask=1
            self.kTest=keyTest
            self.ktBrain=keyTestBrain
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
            self.kBrainDC=2*self.tp/(self.kBrain.shape[0]+self.ktBrain.shape[0])
            self.kSkullDC=2*self.tn/(self.kSkull.shape[0]+self.ktSkull.shape[0])
        elif np.any(keyBrain) and np.any(keySkull):
            self.hasMask=0
            self.kTest=keyTest
            self.ktBrain=keyTestBrain
            self.ktSkull=SubstractKeyImages(self.kTest,self.ktBrain)
            self.kBrain=keyBrain
            self.kSkull=keySkull
            self.tp=CompareKeyImages(self.kBrain,self.ktBrain)
            self.fp=CompareKeyImages(self.kBrain,self.ktSkull)
            self.tn=CompareKeyImages(self.kSkull,self.ktSkull)
            self.fn=CompareKeyImages(self.kSkull,self.ktBrain)
            self.brainAccuracy=self.tp/(self.tp+self.fp)
            self.notBrainAccuracy=self.tn/(self.tn+self.fn)
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
            print('predicted brain: ',self.kBrain.shape[0])
            print('predicted notBrain: ',self.kSkull.shape[0])
            print('|groundTruth Brain|: ',self.ktBrain.shape[0])
            print('brain Accuracy: ',self.brainAccuracy)
            print('notBrain Accuracy: ',self.notBrainAccuracy)
            print('key brain dc: ',self.kBrainDC)
            print('key notBrain dc: ',self.kSkullDC)
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