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
nB=30
i=0
start=time.time()
allMatches=kt.keypointDescriptorMatch(allKey[0],allKey[1:nB])


testImage=allKey[i]
trainingImages=allKey[i+1:nB]

listMatches=kt.matchDistanceSelection(allMatches,testImage,trainingImages)

allKeyFiles=ut.getListFileKey(commonPath)
allAsegPaths=ut.getAsegPaths(allKeyFiles)
trainingAsegPaths=allAsegPaths[i+1:]
asegTestPath=allAsegPaths[i]
listLabels=kt.getAllLabels(trainingAsegPaths,listMatches,trainingImages)

pMap,mLL=kt.voting2(testImage,trainingImages,listMatches,listLabels)

allBrainPaths=ut.getBrainPath(allKeyFiles)
trainingBrainPaths=allBrainPaths[i+1:]
testBrain=kt.getNiiData(allBrainPaths[0])
asegTest=kt.getNiiData(asegTestPath)

segMap,lMap=kt.doSeg2(testImage,listMatches,mLL,trainingImages,trainingAsegPaths,trainingBrainPaths,testBrain,pMap,listLabels)

end=time.time()
print(end-start)
uniqueLabel=np.unique(mLL)
truth=kt.getNiiData(allAsegPaths[0])
for j in range(uniqueLabel.shape[0]):
    print(ut.getDC(segMap,truth,uniqueLabel[j]))
    