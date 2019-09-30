import numpy as np
from pyflann import *
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import nibabel as nib
import utilities as ut
from numba import guvectorize,float64,int64
from numpy import linalg as LA


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

def getSubArrayByRadius(array,XYz,radius):
    """
    return a cubic or square array centered on XYz and with radius*2+1 size
     *** INPUT ***
    array: array to get the subArray from
    XYz: center of the sub array, either [x,y,z] or [x,y] depending on the dimension of array
    radius: radius of the sub array. Dimension of the subArray will be (r*2+1)^(2or3)
     *** OUTPUT ***
    output:subArray
    """
    if len(array.shape)>2:
        return array[XYz[0]-radius:XYz[0]+radius+1,XYz[1]-radius:XYz[1]+radius+1,XYz[2]-radius:XYz[2]+radius+1]
    else:
        return array[XYz[0]-radius:XYz[0]+radius+1,XYz[1]-radius:XYz[1]+radius+1]

def getNiiData(niiPath):
    """
    get data from a nii file
     *** INPUT ***
    niiPath: path of the nii file
     *** OUTPUT ***
    return data (in our case a 3D numpy array)
    """
    img=nib.load(niiPath)
    return np.float32(img.get_fdata())
    
    

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

def getAllLabels(niiPaths,allMatches,keyTrainingData):
    """
    loop of getImgLabels over all keypoints contained in allMatches
     *** INPUT ***
    niipaths: niipaths of all keyTrainingData in the same order as keyTrainingData
    allMatches: list of matches. Same lenght and order as keyTrainingData.
                allMatches[x] returns an array of 3 colums
                testImage key - trainingImage key matched to it - probability
    keyTrainingData: list of all keyTrainingData
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
        trainingImage=keyTrainingData[img]
        XYZ=(trainingImage[matches[:,1].astype(int),kIP.XYZ]).astype(int)
        listLabels.append(getImgLabels(niiPath,XYZ))
    return listLabels
        
        
        
def keypointDescriptorMatch(keyTest, keyTrainingData):
    """
    for each training image
        for each keypoint in test image
            finds the closest keypoint in the training image and approve it or not
     *** INPUT ***
    keyTest: SIFT keypoint of the test image
    keyTrainingData: list containing SIFT keypoints for all keyTrainingData
     *** OUTPUT ***
    list of matches, in the same order as keyTrainingData
    """
    #create one matchmap for each training images
    #for each keypoint in keyTest
        #find if there is any similar keypoint in the trainingImage
    flannM = FLANN()
    allMatches=[]
    for imageI in range(len(keyTrainingData)):
        
        trainingImage=keyTrainingData[imageI]
        params = flannM.build_index(trainingImage[:,kIP.descriptor], algorithm="kdtree",trees=4);
        nbKey=keyTest.shape[0]
        matches=np.full((nbKey,2),-1)
        
        for keyI in range(nbKey):
            #get 2 closest neighbors
            result, dist = flannM.nn_index(keyTest[keyI,kIP.descriptor],2, checks=params["checks"]);
            
            #check scale and distance ratio
            sT=trainingImage[result[0,0],kIP.scale]
            sI=keyTest[keyI,kIP.scale]
            if (sI/sT>=0.5 and sI/sT<=2) and (dist[0,0]/dist[0,1]<=0.9):
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
    s=2*maxDiff+1
    probMap=np.zeros((s,s,s),dtype=np.float64)
    for j in range(np.shape(dXYZ)[0]):
            xyz= tuple((np.floor(dXYZ[j,:])+maxDiff).astype(int))
            probMap[xyz]=probMap[xyz]+1
    probMap=gaussian_filter(probMap,sigma)
    probableDXYZ=np.unravel_index(np.argmax(probMap),probMap.shape)-maxDiff
    
    return [probMap, probableDXYZ]

def matchDistanceSelection(allMatches,keyTest,keyTrainingData):
    """
    __FUNCTION DESCRIPTION__
    Use the distance difference between matches to evaluate how likely that match is. 
    Then trim down the list of matches based on that.
    *** INPUT ***
    allMatches: output from keypointDescriptorMatch funciton
    keyTest: SIFT keypoint of the test image
    keyTrainingData: list containing SIFT keypoints for all keyTrainingData
     *** OUTPUT ***
    trimmed down list of matches
    """
    distanceTestedMatches=[]
    for j in range(len(allMatches)):
        matches=allMatches[j]
        keyTrainingDatum=keyTrainingData[j]
        
        testXYZ=keyTest[matches[:,0],kIP.XYZ]
        trainingXYZ=keyTrainingDatum[matches[:,1],kIP.XYZ]
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

def voting(keyTest,keykeyTrainingData,listMatches,listLabels,nbLabel=41000):
    """
    Uses match probability and keypoint similarity to estimate the most likeyly
    label for each keypoint in the test image
     *** INPUT ***
    keyTest: SIFT keypoint of the test image
    keykeyTrainingData: list containing SIFT keypoints for all keykeyTrainingData
    listMatches: output from matchDistanceSelection funciton
    listLabels: output of getAllLabels(trainingAsegPaths,listMatches,keykeyTrainingData)
    nbLabel: number of possible labels in the images
     *** OUTPUT ***
    pMap: array containing for each test keypoint label combination [nbLabel,nbKeyTest]
    mLL: most likeyly label for each test keypoint [nbKeyTest]
    """
#    maxLabel=0
    nbkeykeyTrainingData=len(keykeyTrainingData)
    nbKeyTest=np.shape(keyTest)[0]
    pMap=np.zeros((nbLabel,nbKeyTest))
    
    
    for k in range(nbKeyTest):
        descriptorDistances=np.zeros((nbkeykeyTrainingData))
        matchProb=np.zeros((nbkeykeyTrainingData))
        labels=np.zeros((nbkeykeyTrainingData))
        tauxTaux=0
        for i in range(nbkeykeyTrainingData):  
            
            keyTrainingDatum=keykeyTrainingData[i]
            matches=listMatches[i]
            idxMatchedTestedKey=matches[:,0]==k
            if np.sum(idxMatchedTestedKey)>0:
                matchProb[i]=matches[idxMatchedTestedKey,2]
                matchedTestedKey=keyTest[k,:]
                idxMatchedTrainingKey=matches[idxMatchedTestedKey,1].astype(int)
                matchedTrainingKey=np.squeeze(keyTrainingDatum[idxMatchedTrainingKey,:])
                #calculate euclidean distance without squaring the result
                descriptorDistances[i]=np.sum(np.power(matchedTestedKey[kIP.descriptor]-matchedTrainingKey[kIP.descriptor],2))               
                keyTrainingDatumLabels=listLabels[i]              
                labels[i]=keyTrainingDatumLabels[idxMatchedTestedKey]
                
        tauxTaux=np.power(np.max(descriptorDistances),2)

        for i in range(nbkeykeyTrainingData):
            #if tauxTaux==0 it means no match to k where found
            if tauxTaux>0 and matchProb[i]>0:
                #keyPointProbability2=(1/np.sqrt(2*np.pi*tauxTaux))*np.exp(-descriptorDistances[i]/(2*tauxTaux)) #non log variant
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
    """
    Create a 3D array containing the distance of each voxel the xyz input
     *** INPUT ***
    shape=shape of the created array
    xyz= voxel to calculate distance from in the array
     *** OUTPUT ***
    distance: distance array
    """
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

@guvectorize([(float64[:,:,:],float64[:,:,:],int64[:], int64,int64[:])], '(x,y,z),(x2,y2,z2),(m),()->(m)',nopython=True)
def findBestPatchMatch(image,patch,target,radius,bestMatch):
    """
    Uses keypoint match as approximation and then finds the best correspondance using patch comparison
     *** INPUT ***
    image: volumetric test image
    patch: patch from the training data
    target: XYZ coordinate of the match on the image, always matched to the center of the patch
    radius: radius of search
    bestMatch: initialized array that will be used as a output
     *** OUTPUT ***
    bestMatch: new xyz for the match on image
    """
    pR=(patch.shape[0]-1)/2
    bestDiff=0
    for x in range(target[0]-radius,target[0]+radius+1):
        for y in range(target[1]-radius,target[1]+radius+1):
            for z in range(target[2]-radius,target[2]+radius+1):
                imageP=image[x-pR:x+pR+1,y-pR:y+pR+1,z-pR:z+pR+1]
                imagePAvg=np.mean(imageP)
                patchAvg=np.mean(patch)
                diff=1/(1+np.sum(np.abs((imageP-imagePAvg)-(patch-patchAvg))))
                if diff > bestDiff:
                    bestDiff=diff
                    bestMatch[:]=[x,y,z]
                    
def ComparePatchDOT(p1,p2):
    p1Avg=np.average(p1)
    p2Avg=np.average(p2)
    np1=p1.flatten()
    np2=p2.flatten()
    t1=np1-p1Avg
    t2=np2-p2Avg
    np1=(t1)/LA.norm(t1)
    np2=(t2)/LA.norm(t2)
    dotP=np.dot(np1,np2)
    if np.isnan(dotP):
        print('nan in comparePatch')
        print(np.average(t1),   LA.norm(t1))
    return (1-dotP)

def ComparePatchSUM(p1,p2,v1,v2):
    p1Avg=np.average(p1)
    p2Avg=np.average(p2)
    p1=np.power(p1-p1Avg,2)
    p2=np.power(p2-p2Avg,2)
    c=np.sum(p1)/(v1*p1.size)-np.sum(p2)/(v2*p1.size)
    return c
    
    
    
def fill(array2D,y0):
    for x in range(array2D.shape[0]):
           ind=np.argwhere(array2D[x,:])
           if ind.any():
               d=int(y0-ind[0])
               array2D[x,y0-d:y0+d+1]=True
    
def drawCircle(array, x0, y0, radius):
    #mid-point circle drawing algorithm
    f = 1 - radius
    ddf_x = 1
    ddf_y = -2 * radius
    x = 0
    y = radius
    array[x0, y0 + radius]=True
    array[x0, y0 - radius]=True
    array[x0 + radius, y0]=True
    array[x0 - radius, y0]=True

    while x < y:
        if f >= 0: 
            y -= 1
            ddf_y += 2
            f += ddf_y
        x += 1
        ddf_x += 2
        f += ddf_x    
        array[x0 + x, y0 + y ]=True
        array[x0 - x, y0 + y ]=True
        array[x0 + x, y0 - y ]=True
        array[x0 - x, y0 - y ]=True
        array[x0 + y, y0 + x ]=True
        array[x0 - y, y0 + x ]=True
        array[x0 + y, y0 - x ]=True
        array[x0 - y, y0 - x ]=True
def drawSphere(array,x0,y0,z0,radius):
    r=0
    drawCircle(array[:,:,z0],x0,y0,radius)
    fill(array[:,:,z0],y0)
    for i in range(radius,0,-1):  
        
        while radius+0.5>np.sqrt(r**2 +i**2):
            r+=1
        r-=1
        drawCircle(array[:,:,z0+i],x0,y0,r)
        drawCircle(array[:,:,z0-i],x0,y0,r)
        fill(array[:,:,z0+i],y0)
        fill(array[:,:,z0-i],y0)
               
def doSeg(keyTest,listMatches,mLL,keyTrainingData,trainingAsegPaths,trainingVolumePaths,testVolume,pMap,listLabels,outputInfoFile,generateDistanceInfo):
    """
    Transfer segmentation from training images to a probability map based on most likely labels, pixel similarity,
    pMap and the match probability
     *** INPUT ***
    keyTest: SIFT keypoint of the test image
    listMatches: output from matchDistanceSelection funciton
    mLL: most likey labels, output from voting function
    keyTrainingData: list containing SIFT keypoints for all keyTrainingData
    trainingAsegPaths: list of paths of training images segmentation
    trainingVolumePaths: list of paths of training images grescale images
    testVolume: test image in greyscale
    pMap: output of voting funciton
    listLabels: output of getAllLabels(trainingAsegPaths,listMatches,keyTrainingData)
     *** OUTPUT ***
    lMap: segmentaiton map (final)
    lMapProb: label probability for each pixel (for debugging use)
    """
    f=outputInfoFile
    labelList=np.unique(mLL)
    nbLabel=labelList.shape[0]
    imageShape=testVolume.shape
    testVar=np.var(testVolume[testVolume>0])
    lMapProb=np.zeros((imageShape[0],imageShape[1],imageShape[2],nbLabel),dtype=np.float32)
    maxDist=np.max(np.shape(lMapProb))
    distSave=np.zeros((maxDist,2))
    listOfKeyTransfered=np.full((keyTest.shape[0],3),-1)
    for k in range(np.shape(keyTest)[0]):
        
        for i in range(np.shape(listMatches)[0]):
            matches=listMatches[i]
            trainingImage=keyTrainingData[i]
            
            #check if k Key is matched with i training Image
            if np.sum(matches[:,0]==k)>0:
                trainingKeyIndex=int(matches[matches[:,0]==k,1])
                #get Training seg and label of training keypoint
                
                trainingImageLabels=listLabels[i]
                label=trainingImageLabels[matches[:,0]==k].astype(int)

                if mLL[k]==label and label!=0:

                    #get brain map
                    listOfKeyTransfered[k,:]=keyTest[k,0:3]
                    trainingBrain=getNiiData(trainingVolumePaths[i])
                    trainingAseg=getNiiData(trainingAsegPaths[i])
                    
                    segMap=trainingAseg==label
  
                    labelIndex=labelList==label
                    
                    #calculating initial translation
                    XYZt=np.int64(trainingImage[trainingKeyIndex,0:3])
                    XYZ=np.int64(keyTest[k,0:3])

                    #pinpointing best translation
                    pR=np.int64(5)
                    bestMatch=np.zeros(3,dtype=np.int64)
                    patch=getSubArrayByRadius(trainingBrain,XYZt,pR)
                    findBestPatchMatch(testVolume,patch,XYZ,pR,bestMatch)
                    XYZ=bestMatch
                    [dx,dy,dz]=XYZ-XYZt
                    
                    transferedSeg=np.zeros(testVolume.shape)
                    c=1
                    r=2 
                    sPR=3 #subPatchRadius
                    maxRad=30 #temporary
                    patchFilter0=np.zeros((2*maxRad+1,2*maxRad+1,2*maxRad+1),dtype=np.bool)
                    getOut=0
                    bandSize=3
                    cAvg=1
                    intVar=np.var(trainingImage[trainingImage>0])
                    while  cAvg>=0.001 and r<maxRad-bandSize:
                        #check if current patch is out of bound of image, stops if it is
                        for ii in range(3):
                            if (min(XYZ[ii]-r-sPR,XYZt[ii]-r-sPR)<0) or (max(XYZ[ii]+r+sPR,XYZt[ii]+r+sPR)>=imageShape[ii]):
                                getOut=1
                        if getOut==1:
                            break
                        
                        subPatchSegMap=getSubArrayByRadius(segMap,XYZt,r)
                        patchFilter1=np.zeros((2*maxRad+1,2*maxRad+1,2*maxRad+1),dtype=np.bool)
                        drawSphere(patchFilter1,maxRad,maxRad,maxRad,r)
                        patchFilter=np.logical_xor(patchFilter0,patchFilter1)
                        subPatchFilter=getSubArrayByRadius(patchFilter,[maxRad,maxRad,maxRad],r)
                        f=subPatchSegMap*subPatchFilter
                        if np.sum(f)>0:
                            pC=np.argwhere(f) #patch coordinate
                            cSum=0
                            for ii in range(np.sum(f)):
                                #compares 2 patches from testVollume and trainingBrain
                                #index goes to XYZ (point of comparison) -> - radius of subPatchFilters -> + coordinate in those patch 
                                #then make a cube around this point of radius sPR
                                c=ComparePatchSUM(testVolume[XYZ[0]-r+pC[ii,0]-sPR:XYZ[0]-r+pC[ii,0]+sPR+1,
                                                          XYZ[1]-r+pC[ii,1]-sPR:XYZ[1]-r+pC[ii,1]+sPR+1,
                                                          XYZ[2]-r+pC[ii,2]-sPR:XYZ[2]-r+pC[ii,2]+sPR+1],
                                               trainingBrain[XYZt[0]-r+pC[ii,0]-sPR:XYZt[0]-r+pC[ii,0]+sPR+1,
                                                             XYZt[1]-r+pC[ii,1]-sPR:XYZt[1]-r+pC[ii,1]+sPR+1,
                                                             XYZt[2]-r+pC[ii,2]-sPR:XYZt[2]-r+pC[ii,2]+sPR+1],testVar,intVar) 
                                c=(1/np.sqrt(2*np.pi*intVar))*np.exp(-c*c/(2*intVar))
                                cSum+=c
                                transferedSeg[XYZ[0]-r+pC[ii,0],XYZ[1]-r+pC[ii,1],XYZ[2]-r+pC[ii,2]]\
                                =c+transferedSeg[XYZ[0]-r+pC[ii,0],XYZ[1]-r+pC[ii,1],XYZ[2]-r+pC[ii,2]]
                            cAvg=cSum/np.sum(f)
                        patchFilter0=patchFilter1
                        r+=bandSize
                                                                
#                        if generateDistanceInfo==0:
#                            #measuring intensity in function of distance
#                            distArray=createDistanceArray(toAdd.shape,[xT,yT,zT])
#                            for j in range(maxDist):
#                                n=np.sum(distArray==j)
#                                if n>0:
#                                    distSave[j,0]=+np.sum(temp[distArray==j])
#                                    distSave[j,1]=+n  

                    lMapProb[:,:,:,labelIndex]+=np.expand_dims(transferedSeg,axis=3)
        print("seg keypoint nb\t",k,'/',keyTest.shape[0])

    lMap=np.zeros(imageShape)  
#    maxProb=np.max(lMapProb)  
#    lMapProb[lMapProb[:,:,:,0]>0.15*maxProb]=0.15*maxProb    
    lMap1=np.argmax(lMapProb,axis=3)
    

    for i in range(nbLabel):
        binaryMap=lMap1==i
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
        