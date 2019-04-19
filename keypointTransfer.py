import numpy as np
from pyflann import *

class keyInformationPosition():
    scale=3
    XYZ=slice(0,3,1)#np.arange(0,3)
    descriptor=slice(16,81,1)#np.arange(16,81)
    
kIP=keyInformationPosition()


        
def keypointDescriptorMatch(testImage, trainingImages):
    #create one matchmap for each training images
    #for each keypoint in testImage
        #find if there is any similar keypoint in the trainingImage
    flannM = FLANN()
    allMatches=[]
    for imageI in range(np.shape(trainingImages)[0]):
        
        trainingImage=trainingImages[imageI]
        params = flannM.build_index(trainingImage[:,kIP.descriptor], algorithm="autotuned", target_precision=0.9, log_level = "info");
        
        matches=np.full((np.shape(testImage)[0],2),-1)
        
        for keyI in range(np.shape(testImage)[0]):
            #get 2 closest neighbors
            result, dist = flannM.nn_index(testImage[keyI,kIP.descriptor],2, checks=params["checks"]);
            
            #check scale and distance ratio
            sT=trainingImage[result[0,1],kIP.scale]
            sI=testImage[keyI,kIP.scale]
            if (sI/sT>=0.5 and sI/sT<=2) and dist[0,0]/dist[0,1]<=0.9:
                matches[keyI,:]=[keyI,result[0,1]]
        #remove all key row where there were no valid matches
        matches2=matches[matches.min(axis=1)>=0,:]
        allMatches.append(matches2)
        print("index of image matched:",imageI)
    return allMatches   

def houghTransform(matchedXYZDifference):
    #input should be matchedTestXYZ-matchedtrainingXYZ
    dXYZ=matchedXYZDifference
    flannH=FLANN()
    params = flannH.build_index(dXYZ, algorithm="kdtree");
    nbPointInSphere=np.round(np.shape(dXYZ)[0]/2).astype(int)
    
    #initiation of bestDist and bestKeyI
    result, dist = flannH.nn_index(dXYZ[0,:],nbPointInSphere, checks=params["checks"]);
    distSum=np.sum(dist)
    bestDist=distSum
    bestKeyI=0
    
    for keyI in range(1,np.shape(dXYZ)[0]):
        result, dist = flannH.nn_index(dXYZ[keyI,:],nbPointInSphere, checks=params["checks"]);
        distSum=np.sum(dist)
        if distSum<bestDist:
            bestDist=distSum
            bestKeyI=keyI
    probableDXYZ=dXYZ[bestKeyI,:]
    return probableDXYZ

def getMatchProbabilityDistribution(distanceTestedXYZDifference):
    n=np.shape(distanceTestedXYZDifference)[0]
    sumOfDistance=np.zeros(n)
    probabilityDistribution=np.ones(n)
    
    for i in range(np.shape(distanceTestedXYZDifference)[0]):
        d=np.copy(distanceTestedXYZDifference)
        d=d-d[i,:]
        d=np.multiply(d,d)
        sumOfDistance[i]=np.sum(d)
    
    
    probabilityDistribution=np.divide(probabilityDistribution,sumOfDistance)
    probabilityDistribution=probabilityDistribution/(np.sum(probabilityDistribution))
    
    return probabilityDistribution    
        

def matchDistanceSelection(allMatches,testImage,trainingImages):
    distanceTestedMatches=[]
    for j in range(len(allMatches)):
        match=allMatches[j]
        trainingImage=trainingImages[j]
        t=np.zeros((np.shape(match)[0],3))
        
        testXYZ=testImage[match[:,0],kIP.XYZ]
        trainingXYZ=trainingImage[match[:,1],kIP.XYZ]
        matchedXYZDifference=testXYZ-trainingXYZ
        probableTranslation=houghTransform(matchedXYZDifference)
        
        #XYZtest-XYZtraining-translation
        tMeanMatrix=np.zeros((np.shape(match)[0],3))
        tMeanMatrix[:,:]=probableTranslation
        XYZT=t-tMeanMatrix
        
        distance=np.sum(np.absolute(XYZT),axis=1)
        indSort=np.argsort(distance)
        nbMatchKeep=round(np.shape(match)[0]/10)
        outMatch=np.zeros((nbMatchKeep,3))
        outMatch[:,0:2]=match[indSort[0:nbMatchKeep],:]
        outMatch[:,2]=getMatchProbabilityDistribution(matchedXYZDifference[indSort[0:nbMatchKeep],:])
        distanceTestedMatches.append(outMatch)
        
    return distanceTestedMatches

def voting2(testImage,trainingImages,listMatches,trainingAsegPaths):
#    maxLabel=0
    nbLabel=256
    nbKeyTest=np.shape(testImage)[0]
    nbTrainingImages=len(trainingImages)
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
                matchedTrainingKey=np.squeeze(trainingImage[matches[matches[:,0]==k,0].astype(int),:])
                #calculate euclidean distance without squaring the result
                descriptorDistances[i]=np.sum(np.power(matchedTestedKey[kIP.descriptor]-matchedTrainingKey[kIP.descriptor],2))
                trainingAseg=getAllSiftData.getNiiData(trainingAsegPaths[i])
                trainingKey=matches[matches[:,0]==k,1].astype(int)
                XYZ=trainingImage[trainingKey,kIP.XYZ].astype(int)
                XYZ=XYZ[0]
                labels[i]=trainingAseg[XYZ[0],XYZ[1],XYZ[2]]
                
        tauxTaux=np.power(np.max(descriptorDistances),2)
        for i in range(nbTrainingImages):
            if tauxTaux>0:
                keyPointProbability=(1/np.sqrt(2*np.pi*tauxTaux))*np.exp(-descriptorDistances[i]/(2*tauxTaux))
               
                label=labels[i].astype(int)
                pMap[label,k]=pMap[label,k]+keyPointProbability*pm[i]
                    
    return pMap

def mostLikelyLabel2(pMap):
    mLL=np.zeros(np.shape(pMap)[1])
    for k in range(np.shape(pMap)[1]):
        i=np.argmax(np.squeeze(pMap[:,k]))
        mLL[k]=i

def doSeg2(testImage,listMatches,mLL,trainingImages,trainingAsegPaths,trainingBrainPaths,testBrain,pMap):
    lMap=np.zeros((256,256,256,256),dtype=np.float16)
    
    for k in range(np.shape(testImage)[0]):
        
        for i in range(np.shape(listMatches)[0]):
            matches=listMatches[i]
            
            #check if k Key is matched with i training Image
            if np.sum(matches[:,0]==k)>0:
                #get Training seg and label of training keypoint
                trainingAseg=getAllSiftData.getNiiData(trainingAsegPaths[i])
                trainingImage=trainingImages[i]
                trainingKey=matches[matches[:,0]==k,1].astype(int)
                XYZ=trainingImage[trainingKey,kIP.XYZ][0].astype(int)
                label=trainingAseg[XYZ[0],XYZ[1],XYZ[2]].astype(int)
                
                if mLL[k]==label and label!=0:
                    #get brain map
                    trainingBrain=getAllSiftData.getNiiData(trainingBrainPaths[i])
                    
                    segMap=trainingAseg==label
                    
                    iTest=testBrain*segMap
                    iTraining=trainingBrain*segMap
                    iDiff=iTest-iTraining

                    v=np.var(iDiff) #to confirm
                    if v==0:
                        print("pause")
                    c=1/np.sqrt(2*np.pi*v)
                    
                    #a checker
                    W=c*np.exp(-np.power(iDiff,2)/(2*np.power(v,2)))
                    
                    #pmap [nbLabel,nbKeyTest,nbTrainingImages]
                    lMap[:,:,:,label]=lMap[:,:,:,label]+W*pMap[label,k]*matches[matches[:,0]==k,2]
    lMapFinal=np.zeros((256,256,256))              
    lMapFinal=np.argmax(lMap,axis=3)
    return lMapFinal