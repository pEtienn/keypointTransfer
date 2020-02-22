# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:23:09 2020

@author: Etenne Pepyn
"""
def KeepBestRotation(pK,testKey):
    #lowers the DC when used with space normalisation
    newpK=np.empty((0,pK.shape[1]))
    newTestKey=np.empty((0,testKey.shape[1]))
    tempP=np.zeros((1,2))
    startIndex=0
    previousXYZ=testKey[0,kIP.XYZ]
    for i in range(pK.shape[0]+1):
        if i==pK.shape[0] or np.all(previousXYZ!=testKey[i,kIP.XYZ]):
            maxIdx=np.argmax(pK[startIndex:i,:],axis=0)
            maxIdx=maxIdx+startIndex
            tempP=np.array([pK[maxIdx[0],0],pK[maxIdx[1],1]])
            newpK=np.append(newpK,[tempP],axis=0)
            newTestKey=np.append(newTestKey,[testKey[maxIdx[0],:]],axis=0) # cant use descriptors after that
            startIndex=i
        if i!=pK.shape[0]:
           previousXYZ=testKey[i,kIP.XYZ]       
    
    return [newpK,newTestKey]

def CombineDescriptorsOfKeyFiles_deprecated(srcPath,dstPath):
    allKeys=os.listdir(srcPath)
    allDes=np.empty((0,64),dtype=np.int8)
    for keyFile in allKeys:
        keys=ReadKeypoints(os.path.join(srcPath,keyFile))
        des=np.int8(keys[:,kIP.descriptor])
        allDes=np.append(allDes,des,axis=0)
        
    pandas.DataFrame(allDes).to_csv(dstPath+'.des', header=None, index=None,sep='\t')
    
def  SelectKeyRotation_deprecated(folderPath,rotation=True):
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

def CreateCombinedMask_deprecated(maskPaths,varPerMatches,dXYZ,resolutionTest,brainMask=True,maskToKeep=5):
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

def UnitTestGenerateTrainingDatabase_old(databasePath,srcKeyPath):
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

def RemoveTestPatientFromFilePaths(listTestNumber,brainFilePaths,skullFilePaths):
    for testNumber in listTestNumber:
        brainFilePaths=[x for x in brainFilePaths if testNumber not in os.path.basename(x)]
        skullFilePaths=[x for x in skullFilePaths if testNumber not in os.path.basename(x)]
    return [brainFilePaths, skullFilePaths]

