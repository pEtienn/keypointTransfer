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
nB=10
i=0
start=time.time()
aM=kt.keypointDescriptorMatch(allKey[0],allKey[1:nB])


testImage=allKey[i]
trainingImages=allKey[i+1:nB]

listMatches=kt.matchDistanceSelection(aM,allKey[0],allKey[1:nB])
allKeyFiles=ut.getListFileKey(commonPath)
allAsegPaths=ut.getAsegPaths(allKeyFiles)
trainingAsegPaths=allAsegPaths[i+1:]
asegTestPath=allAsegPaths[i]
pMap=kt.voting2(testImage,trainingImages,listMatches,trainingAsegPaths)
mLL=kt.mostLikelyLabel2(pMap)

allBrainPaths=ut.getBrainPath(allKeyFiles)
trainingBrainPaths=allBrainPaths[i+1:]
testBrain=ut.getNiiData(allBrainPaths[0])
asegTest=ut.getNiiData(asegTestPath)
a=kt.doSeg2(testImage,listMatches,mLL,trainingImages,trainingAsegPaths,trainingBrainPaths,testBrain,pMap)

end=time.time()
print(end-start)