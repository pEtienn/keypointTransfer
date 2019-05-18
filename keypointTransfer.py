import numpy as np
from pyflann import *
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import nibabel as nib
import time

#COMMENT TEMPLATE
"""
__FUNCTION DESCRIPTION__
 *** INPUT ***
__list here__
 *** OUTPUT ***
__list here__
"""

class keyInformationPosition():
    scale=3
    XYZ=slice(0,3,1)#np.arange(0,3)
    descriptor=slice(16,81,1)#np.arange(16,81)
    
kIP=keyInformationPosition()

def getNiiData(niiPath):
    img=nib.load(niiPath)
    return img.get_fdata()

def getImgLabels(niiPath,keypointXYZs):
    img=getNiiData(niiPath)
    labelsOut=np.zeros(keypointXYZs.shape[0])
    for key in range(keypointXYZs.shape[0]):
        xyz=tuple((keypointXYZs[key,:]).astype(int))
        labelsOut[key]=img[xyz]
    return labelsOut

def getAllLabels(niiPaths,allMatches,trainingImages):
    listLabels=[]
    for img in range(len(allMatches)):
        matches=allMatches[img]
        niiPath=niiPaths[img]
        trainingImage=trainingImages[img]
        XYZ=(trainingImage[matches[:,1].astype(int),kIP.XYZ]).astype(int)
        listLabels.append(getImgLabels(niiPath,XYZ))
    return listLabels
        
        
        
def keypointDescriptorMatch(testImage, trainingImages):
    #create one matchmap for each training images
    #for each keypoint in testImage
        #find if there is any similar keypoint in the trainingImage
    flannM = FLANN()
    allMatches=[]
    for imageI in range(len(trainingImages)):
        
        trainingImage=trainingImages[imageI]
        params = flannM.build_index(trainingImage[:,kIP.descriptor], algorithm="kdtree",trees=4);
        
        matches=np.full((np.shape(testImage)[0],2),-1)
        
        for keyI in range(np.shape(testImage)[0]):
            #get 2 closest neighbors
            result, dist = flannM.nn_index(testImage[keyI,kIP.descriptor],2, checks=params["checks"]);
            
            #check scale and distance ratio
            sT=trainingImage[result[0,0],kIP.scale]
            sI=testImage[keyI,kIP.scale]
            if (sI/sT>=0.5 and sI/sT<=2) and dist[0,0]/dist[0,1]<=0.9:
                matches[keyI,:]=[keyI,result[0,0]]
        #remove all key row where there were no valid matches
        matches2=matches[matches.min(axis=1)>=0,:]
        allMatches.append(matches2)
        print("index of image matched:",imageI)
    return allMatches   

def houghTransformGaussian(matchedXYZDifference,sigma=3):
    """
    hough transform using using gaussian filter instead of bins to detect most probable translation
     *** INPUT ***
    matchedXYZDifference: x*y
       x:number of match
       y: X Y Z
    sigma: parameter of the gaussian filter
     *** OUTPUT ***
    probMap: probability map indexed by translation XYZ
    probableDXYZ: most probable translation
    """
    dXYZ=matchedXYZDifference
    maxDiff=np.amax(np.abs(dXYZ)).astype(int)
    s=maxDiff*2+1
    probMap=np.zeros((s,s,s))
    for j in range(np.shape(dXYZ)[0]):
        xyz= tuple((np.floor(dXYZ[j,:])+maxDiff).astype(int))
        probMap[xyz]=probMap[xyz]+1
    probableDXYZ=np.unravel_index(np.argmax(probMap),probMap.shape)-maxDiff
    probMap=gaussian_filter(probMap,sigma)
    #probMap=probMap/np.sum(probMap)  
    
    return [probMap, probableDXYZ]

def matchDistanceSelection(allMatches,testImage,trainingImages):
    distanceTestedMatches=[]
    for j in range(len(allMatches)):
        match=allMatches[j]
        trainingImage=trainingImages[j]
        
        testXYZ=testImage[match[:,0],kIP.XYZ]
        trainingXYZ=trainingImage[match[:,1],kIP.XYZ]
        matchedXYZDifference=testXYZ-trainingXYZ
        [probMap,probableTranslation]=houghTransformGaussian(matchedXYZDifference)
        
        #XYZtest-XYZtraining-translation
        tMeanMatrix=np.zeros((np.shape(match)[0],3))
        tMeanMatrix[:,:]=probableTranslation
        XYZT=matchedXYZDifference-tMeanMatrix #erreurrrrrrrrrrrrrrrrr
        
        distance=np.sum(np.absolute(XYZT),axis=1)
        indSort=np.argsort(distance)
        nbMatchKeep=round(np.shape(match)[0]/10)
        outMatch=np.zeros((nbMatchKeep,3))
        outMatch[:,0:2]=match[indSort[0:nbMatchKeep],:]
        #to improve: try to remove for loop
        for i in range(nbMatchKeep):
            translation=tuple((matchedXYZDifference[indSort[i],:]+probMap.shape[0]//2).astype(int))
            outMatch[i,2]=probMap[translation]
        #outMatch[:,2]=getMatchProbabilityDistribution(matchedXYZDifference[indSort[0:nbMatchKeep],:])
        distanceTestedMatches.append(outMatch)
        
    return distanceTestedMatches

def voting2(testImage,trainingImages,listMatches,listLabels):
#    maxLabel=0
    nbTrainingImages=len(trainingImages)
    nbLabel=256
    nbKeyTest=np.shape(testImage)[0]
    pMap=np.zeros((nbLabel,nbKeyTest))
    
    
    for k in range(nbKeyTest):
        descriptorDistances=np.zeros((nbTrainingImages))
        pm=np.zeros((nbTrainingImages))
        labels=np.zeros((nbTrainingImages))
        for i in range(nbTrainingImages):  
            
            trainingImage=trainingImages[i]
            matches=listMatches[i]
            if np.sum(matches[:,0]==k)>0:
                pm[i]=matches[matches[:,0]==k,2]
                matchedTestedKey=testImage[k,:]
                idx=matches[matches[:,0]==k,1].astype(int)
                matchedTrainingKey=np.squeeze(trainingImage[idx,:])
                #calculate euclidean distance without squaring the result
                descriptorDistances[i]=np.sum(np.power(matchedTestedKey[kIP.descriptor]-matchedTrainingKey[kIP.descriptor],2))               
                trainingImageLabels=listLabels[i]              
                labels[i]=trainingImageLabels[matches[:,0]==k]
                
        tauxTaux=np.power(np.max(descriptorDistances),2)
        if tauxTaux==0:
             tauxTaux=0.1
        for i in range(nbTrainingImages):
            if tauxTaux>0:
                keyPointProbability=(1/np.sqrt(2*np.pi*tauxTaux))*np.exp(-descriptorDistances[i]/(2*tauxTaux))
               
                label=labels[i].astype(int)
                pMap[label,k]=pMap[label,k]+keyPointProbability*pm[i]
        
    #get most likely labels      
    mLL=np.zeros(np.shape(pMap)[1])
    for k in range(np.shape(pMap)[1]):
        i=np.argmax(np.squeeze(pMap[:,k]))
        mLL[k]=i
    return [pMap,mLL]


def doSeg2(testImage,listMatches,mLL,trainingImages,trainingAsegPaths,trainingBrainPaths,testBrain,pMap,listLabels):
    labelList=np.unique(mLL)
    nbLabel=labelList.shape[0]
    lMap=np.zeros((256,256,256,nbLabel),dtype=np.float64)
    
    for k in range(np.shape(testImage)[0]):
        
        for i in range(np.shape(listMatches)[0]):
            matches=listMatches[i]
            
            #check if k Key is matched with i training Image
            if np.sum(matches[:,0]==k)>0:
                #get Training seg and label of training keypoint
                
                trainingImageLabels=listLabels[i]
                label=trainingImageLabels[matches[:,0]==k].astype(int)
                
                if mLL[k]==label and label!=0:
                    #get brain map
                    trainingBrain=getNiiData(trainingBrainPaths[i])
                    trainingAseg=getNiiData(trainingAsegPaths[i])
                    
                    segMap=trainingAseg==label
                    
                    intensityTest=testBrain*segMap
                    intensityTraining=trainingBrain*segMap
                    intensityDiff=intensityTest-intensityTraining

                    v=np.var(intensityDiff) #to confirm
                    if v==0:
                        v=1 #to test segmentation on same image
                    c=1/np.sqrt(2*np.pi*v)
                    
                    #a checker
                    W=c*np.exp(-np.power(intensityDiff,2)/(2*np.power(v,2)),where=segMap)
                    
                    #pmap [nbLabel,nbKeyTest,nbTrainingImages]
                    labelIndex=labelList==label
                    toAdd=W*pMap[label,k]*matches[matches[:,0]==k,2]
                    temp=np.expand_dims(toAdd,axis=3)
                    lMap[:,:,:,labelIndex]=+temp
                    print("seg keypoint nb\t",k)
    lMapFinal=np.zeros((256,256,256))              
    lMapFinal=np.argmax(lMap,axis=3)
    for i in range(nbLabel):
        binaryMap=lMapFinal==i
        lMapFinal[binaryMap]=labelList[i]
    return [lMapFinal,lMap]