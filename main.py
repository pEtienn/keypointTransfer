import numpy as np
import time
import keypointTransfer as kt
import utilities as ut
import pickle



picklePath="S:/seg sift transfer/allKeys.pickle"
with open(picklePath,'rb') as f:
    allKey=pickle.load(f)
commonPath="S:/ABIDE/preprocessed/"

#allKeyfiles=ut.getListFileKey(commonPath)
#allKey=[]
#for file in allKeyfiles:
#    allKey.append(ut.getDataFromOneFile(file))
#    
#with open('allKeys.pickle','wb') as f:
#    pickle.dump(allKey,f)
    
print("program starting")
testNb=10
start=0
end=20
v1=np.arange(0,len(allKey))
fArray=v1<0
idxTrainingBool=np.copy(fArray)
idxTrainingBool[start:end]=1
idxTrainingBool[testNb]=0
idxTrainingArray=v1[idxTrainingBool]

trainingImages=kt.getSubListFromArrayIndexing(allKey,idxTrainingArray)

testImage=allKey[testNb]


#get all necessary data
allKeyFiles=ut.getListFileKey(commonPath)
allAsegPaths=ut.getAsegPaths(allKeyFiles)
trainingAsegPaths=kt.getSubListFromArrayIndexing(allAsegPaths,idxTrainingArray)
asegTestPath=allAsegPaths[testNb]
allBrainPaths=ut.getBrainPath(allKeyFiles)
trainingBrainPaths=kt.getSubListFromArrayIndexing(allBrainPaths,idxTrainingArray)
testBrain=kt.getNiiData(allBrainPaths[0])
asegTest=kt.getNiiData(asegTestPath)

start=time.time()
allMatches=kt.keypointDescriptorMatch(testImage,trainingImages)
listMatches=kt.matchDistanceSelection(allMatches,testImage,trainingImages)
listLabels=kt.getAllLabels(trainingAsegPaths,listMatches,trainingImages)
pMap,mLL=kt.voting(testImage,trainingImages,listMatches,listLabels)
segMap,lMap=kt.doSeg(testImage,listMatches,mLL,trainingImages,trainingAsegPaths,trainingBrainPaths,testBrain,pMap,listLabels)
end=time.time()

print(end-start)
uniqueLabel=np.unique(mLL)
truth=kt.getNiiData(asegTestPath)
uTruth=np.unique(truth)
result=np.zeros((uTruth.shape[0],4))
result[:,0]=uTruth
for j in range(uTruth.shape[0]):
    result[j,1]=np.sum(truth==uTruth[j])
    result[j,2]=np.sum(segMap==uTruth[j])
    if result[j,2]>0:
        result[j,3]=ut.getDC(segMap,truth,uTruth[j])
        print(result[j,3])
    