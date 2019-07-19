import numpy as np
from pyflann import *
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import nibabel as nib
import utilities as ut

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
    in an array of SIFT keypoints, row are the number of the keypoints, and colums
    contains it's characteristics. Use kIP object to access the right characteristics
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
    loop of getImgLabels over all keypoints contained in allMatches
     *** INPUT ***
    niipaths: niipaths of all trainingImages in the same order as trainingImages
    allMatches: list of matches. Same lenght and order as trainingImages.
                allMatches[x] returns an array of 3 colums
                testImage key - trainingImage key matched to it - probability
    trainingImages: list of all trainingImages
                    contains a list of SIFT key
     *** OUTPUT ***
     list of labels. Each element of the list contains the output of getImgLabels
     for one trainingImage's matches
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
    """
    for each training image
        for each keypoint in test image
            finds the closest keypoint in the training image and approve it or not
     *** INPUT ***
    testImage: SIFT keypoint of the test image
    trainingImages: list containing SIFT keypoints for all trainingImages
     *** OUTPUT ***
    list of matches, in the same order as trainingImages
    """
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
    """
    __FUNCTION DESCRIPTION__
    Use the distance difference between matches to evaluate how likely that match is. 
    Then trim down the list of matches based on that.
    *** INPUT ***
    allMatches: output from keypointDescriptorMatch funciton
    testImage: SIFT keypoint of the test image
    trainingImages: list containing SIFT keypoints for all trainingImages
     *** OUTPUT ***
    trimmed down list of matches
    """
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
    """
    Uses match probability and keypoint similarity to estimate the most likeyly
    label for each keypoint in the test image
     *** INPUT ***
    testImage: SIFT keypoint of the test image
    trainingImages: list containing SIFT keypoints for all trainingImages
    listMatches: output from matchDistanceSelection funciton
    listLabels: output of getAllLabels(trainingAsegPaths,listMatches,trainingImages)
    nbLabel: number of possible labels in the images
     *** OUTPUT ***
    pMap: array containing for each test keypoint label combination [nbLabel,nbKeyTest]
    mLL: most likeyly label for each test keypoint [nbKeyTest]
    """
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
    
def createDistanceArray(shape,xyz):
    s=np.array([shape[0],shape[1],shape[2],3])
    pos=np.zeros(s)
    for j in range(s[0]):
        pos[j,:,:,0]=j
    for j in range(s[1]):
        pos[:,j,:,1]=j
    for j in range(s[2]):
        pos[:,:,j,2]=j
    pos1=pos-xyz
    pos1=np.power(pos1,2)
    distance=np.sum(pos1,axis=3)
    distance=np.sqrt(distance)
    distance=distance.astype(int)
    return distance

def doSeg(testImage,listMatches,mLL,trainingImages,trainingAsegPaths,trainingBrainPaths,testBrain,pMap,listLabels,outputInfoFile,generateDistanceInfo):
    """
    Transfer segmentation from training images to a probability map based on most likely labels, pixel similarity,
    pMap and the match probability
     *** INPUT ***
    testImage: SIFT keypoint of the test image
    listMatches: output from matchDistanceSelection funciton
    mLL: most likey labels, output from voting function
    trainingImages: list containing SIFT keypoints for all trainingImages
    trainingAsegPaths: list of paths of training images segmentation
    trainingBrainPaths: list of paths of training images grescale images
    testBrain: test image in greyscale
    pMap: output of voting funciton
    listLabels: output of getAllLabels(trainingAsegPaths,listMatches,trainingImages)
     *** OUTPUT ***
    lMap: segmentaiton map (final)
    lMapProb: label probability for each pixel (for debugging use)
    """
    f=outputInfoFile
    labelList=np.unique(mLL)
    nbLabel=labelList.shape[0]
    imageShape=testBrain.shape
    lMapProb=np.zeros((256,256,256,nbLabel),dtype=np.float64)
    maxDist=np.max(np.shape(lMapProb))
    distSave=np.zeros((maxDist,2))
    listOfKeyTransfered=np.full((testImage.shape[0],3),-1)
    for k in range(np.shape(testImage)[0]):
        
        for i in range(np.shape(listMatches)[0]):
            matches=listMatches[i]
            
            #check if k Key is matched with i training Image
            if np.sum(matches[:,0]==k)>0:
                trainingKeyIndex=int(matches[matches[:,0]==k,1])
                #get Training seg and label of training keypoint
                
                trainingImageLabels=listLabels[i]
                label=trainingImageLabels[matches[:,0]==k].astype(int)
                
                if mLL[k]==label and label!=0:
                    #get brain map
                    listOfKeyTransfered[k,:]=testImage[k,0:3]
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
                        trainingImage=trainingImages[i]
                        
                        
                        
                        #calculating translation
                        [xT,yT,zT]=trainingImage[trainingKeyIndex,0:3].astype(int)
                        [x,y,z]=testImage[k,0:3].astype(int)
                        [dx,dy,dz]=np.array([x,y,z])-np.array([xT,yT,zT])
                        [sx,sy,sz]=np.max(np.array([[0-dx,0-dy,0-dz],[0,0,0]]),axis=0)
                        [ex,ey,ez]=np.min(np.array([[toAdd.shape[0],toAdd.shape[1],toAdd.shape[2]],[toAdd.shape[0]-dx,toAdd.shape[1]-dy,toAdd.shape[2]-dz]]),axis=0)
                        
                        if generateDistanceInfo==1:
                            #measuring intensity in function of distance
                            distArray=createDistanceArray(toAdd.shape,[xT,yT,zT])
                            for j in range(maxDist):
                                n=np.sum(distArray==j)
                                if n>0:
                                    distSave[j,0]=+np.sum(temp[distArray==j])
                                    distSave[j,1]=+n  
                                
                        lMapProb[sx+dx:ex+dx,sy+dy:ey+dy,sz+dz:ez+dz,labelIndex]=+temp[sx:ex,sy:ey,sz:ez]
        print("seg keypoint nb\t",k,'/',testImage.shape[0])
    lMap=np.zeros((256,256,256))  
    maxProb=np.max(lMapProb)  
    lMapProb[lMapProb[:,:,:,0]>0.15*maxProb]=0.15*maxProb    
    lMap=np.argmax(lMapProb,axis=3)
    

    for i in range(nbLabel):
        binaryMap=lMap==i
        lMap[binaryMap]=labelList[i]
    
    #print distance
    if generateDistanceInfo==1:
        f.write("distance\taverage intensity\n")
        for j in range(maxDist):
            s=str(j)+'\t\t'+str(distSave[j,0])+'\t'+str(distSave[j,1])+'\n'
            f.write(s)
    
    return [lMap,lMapProb,listOfKeyTransfered[listOfKeyTransfered[:,0]>-1]]


#def printAllImages(allAsegPath,outputPath,allKeyFiles,sliceNumber):
#    
#    for i in allAsegPath:
#        seg=getNiiData(i)
#        keypoints=allKeyFiles[i]
#        listOfKeypointCoordinate=keypoint[:,0:3]
#        patientName=
#        ut.generateSagSliceComp(seg,outputPath,listOfKeypointCoordinate,sliceNumber,patientName):
        