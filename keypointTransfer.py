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
    """
    information locations in SIFT datapoints
    """
    scale=3
    XYZ=slice(0,3,1)#np.arange(0,3)
    descriptor=slice(16,81,1)#np.arange(16,81)
    
kIP=keyInformationPosition()

def getSubListFromArrayIndexing(originalList,arrayIndex):
    """
    Use to select non adjacent list elements in a python list, using a numpy array as an index
     *** INPUT ***
    orginalList: python list
    arrayIndex: numpy array 1D, same as if you would index a 1D numpy array with a 1D array index
     *** OUTPUT ***
    list containing all list elements specified by arrayIndex
    """
    outList=[]
    for i in range(arrayIndex.shape[0]):
        outList.append(originalList[arrayIndex[i]])
    return outList

def getNiiData(niiPath):
    """
    get data from a nii file
     *** INPUT ***
    niiPath: path of the nii file
     *** OUTPUT ***
    return data (in our case a 3D numpy array)
    """
    img=nib.load(niiPath)
    return img.get_fdata()

def getImgLabels(niiPath,keypointXYZs):
    """
    Return a list of labels associeted with each keypoints in the segmented image
     *** INPUT ***
    niiPath: path of the segmented image
    keypointXYZs: coordinates of all keypoints
     *** OUTPUT ***
    1D numpy array containing the labels of the keypoints in the order they were provided
    """
    img=getNiiData(niiPath)
    labelsOut=np.zeros(keypointXYZs.shape[0])
    for key in range(keypointXYZs.shape[0]):
        xyz=tuple((keypointXYZs[key,:]).astype(int))
        labelsOut[key]=img[xyz]
    return labelsOut

def getAllLabels(niiPaths,allMatches,trainingImages):
    """
    __FUNCTION DESCRIPTION__
     *** INPUT ***
    __list here__
     *** OUTPUT ***
    __list here__
    """
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
        nbKey=testImage.shape[0]
        matches=np.full((nbKey,2),-1)
        
        for keyI in range(nbKey):
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
    matchedXYZDifference: shape(i,j)
       i:number of match
       j: X Y Z
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
    probMap=gaussian_filter(probMap,sigma)
    probableDXYZ=np.unravel_index(np.argmax(probMap),probMap.shape)-maxDiff
    
    return [probMap, probableDXYZ]

def matchDistanceSelection(allMatches,testImage,trainingImages):
    distanceTestedMatches=[]
    for j in range(len(allMatches)):
        matches=allMatches[j]
        trainingImage=trainingImages[j]
        
        testXYZ=testImage[matches[:,0],kIP.XYZ]
        trainingXYZ=trainingImage[matches[:,1],kIP.XYZ]
        matchedXYZDifference=testXYZ-trainingXYZ
        [probMap,probableTranslation]=houghTransformGaussian(matchedXYZDifference)
        
        probTransMarix=np.full((np.shape(matches)[0],3),probableTranslation)
        XYZT=matchedXYZDifference-probTransMarix #erreurrrrrrrrrrrrrrrrr
        
        distance=np.sum(np.power(XYZT,2),axis=1)
        indSort=np.argsort(distance)
        nbMatchKeep=round(np.shape(matches)[0]/10)
        outMatch=np.zeros((nbMatchKeep,3))
        outMatch[:,0:2]=matches[indSort[0:nbMatchKeep],:]
        for i in range(nbMatchKeep):
            translation=tuple((matchedXYZDifference[indSort[i],:]+probMap.shape[0]//2).astype(int))
            outMatch[i,2]=probMap[translation]
        #outMatch[:,2]=getMatchProbabilityDistribution(matchedXYZDifference[indSort[0:nbMatchKeep],:])
        distanceTestedMatches.append(outMatch)
        
    return distanceTestedMatches

def voting(testImage,trainingImages,listMatches,listLabels,nbLabel=256):
#    maxLabel=0
    nbTrainingImages=len(trainingImages)
    nbKeyTest=np.shape(testImage)[0]
    pMap=np.zeros((nbLabel,nbKeyTest))
    
    
    for k in range(nbKeyTest):
        descriptorDistances=np.zeros((nbTrainingImages))
        matchProb=np.zeros((nbTrainingImages))
        labels=np.zeros((nbTrainingImages))
        tauxTaux=0
        for i in range(nbTrainingImages):  
            
            trainingImage=trainingImages[i]
            matches=listMatches[i]
            idxMatchedTestedKey=matches[:,0]==k
            if np.sum(idxMatchedTestedKey)>0:
                matchProb[i]=matches[idxMatchedTestedKey,2]
                matchedTestedKey=testImage[k,:]
                idxMatchedTrainingKey=matches[idxMatchedTestedKey,1].astype(int)
                matchedTrainingKey=np.squeeze(trainingImage[idxMatchedTrainingKey,:])
                #calculate euclidean distance without squaring the result
                descriptorDistances[i]=np.sum(np.power(matchedTestedKey[kIP.descriptor]-matchedTrainingKey[kIP.descriptor],2))               
                trainingImageLabels=listLabels[i]              
                labels[i]=trainingImageLabels[idxMatchedTestedKey]
                
        tauxTaux=np.power(np.max(descriptorDistances),2)

        for i in range(nbTrainingImages):
            #if tauxTaux==0 it means no match to k where found
            if tauxTaux>0 and matchProb[i]>0:
                keyPointProbability2=(1/np.sqrt(2*np.pi*tauxTaux))*np.exp(-descriptorDistances[i]/(2*tauxTaux))
                keyPointProbability=np.log(1/np.sqrt(2*np.pi*tauxTaux))+(-descriptorDistances[i]/(2*tauxTaux))
                label=labels[i].astype(int)
                temp1=keyPointProbability+np.log(matchProb[i])
                temp11=np.exp(temp1)
#                temp2=keyPointProbability2*matchProb[i]
                pMap[label,k]=pMap[label,k]+temp11
    #get most likely labels      
    mLL=np.zeros(np.shape(pMap)[1])
    for k in range(np.shape(pMap)[1]):
        i=np.argmax(np.squeeze(pMap[:,k]))
        mLL[k]=i
    return [pMap,mLL]


def doSeg(testImage,listMatches,mLL,trainingImages,trainingAsegPaths,trainingBrainPaths,testBrain,pMap,listLabels):
    labelList=np.unique(mLL)
    nbLabel=labelList.shape[0]
    imageShape=testBrain.shape
    lMapProb=np.zeros((256,256,256,nbLabel),dtype=np.float64)
    
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

                    vv=np.var(intensityDiff) #to confirm
                    
                    if vv!=0:
                        #can be 0 if it's a slice without brain in it
                        c=1/np.sqrt(2*np.pi*vv)
                        
                        #a checker
#                        W=c*np.exp(-np.power(intensityDiff,2)/(2*vv),where=segMap)
                        W=np.log(c)+(-np.power(intensityDiff,2)/(2*vv))
                        
                        #pmap [nbLabel,nbKeyTest,nbTrainingImages]
                        labelIndex=labelList==label
#                        toAdd=W*pMap[label,k]*matches[matches[:,0]==k,2]
                        toAdd=np.exp(W+segMap*np.log(pMap[label,k])+segMap*np.log(matches[matches[:,0]==k,2]),where=segMap)
                        temp=np.expand_dims(toAdd,axis=3)
                        lMapProb[:,:,:,labelIndex]=+temp
                        print("seg keypoint nb\t",k)
    lMap=np.zeros((256,256,256))  
    maxProb=np.max(lMapProb)  
    lMapProb[lMapProb[:,:,:,0]>0.15*maxProb]=0.15*maxProb    
    lMap=np.argmax(lMapProb,axis=3)
    

    for i in range(nbLabel):
        binaryMap=lMap==i
        lMap[binaryMap]=labelList[i]
    return [lMap,lMapProb]