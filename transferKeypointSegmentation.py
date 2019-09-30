# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:10:11 2019

@author: Etenne Pepyn
"""

import numpy as np
import keypointTransfer as kt
import utilities as ut
import argparse
import os
import shutil
from datetime import date


"""
Execute the keypoint transfer segmentation algorithm
 *** INPUT ***
 The numbers used for testNb, start and end refers to the position of the 
 file in a list containing all files found in the each of the 3 folders
 ind commonPath.
testNb: number of the test image
start: start of the interval of images used
end: end of the interval of image used
 *** OUTPUT ***
segMap: generated segmentation
the dice coefficients of the segmentation gets printed
"""

if 'f' in locals():
    if f:
        f.close()

parser = argparse.ArgumentParser(description='Execute the keypoint transfer segmentation algorithm')
parser.add_argument('--testFile',help='Name of the test file, default:Caltech_0051456',default='Caltech_0051456')
parser.add_argument('--outputFile',help='Name of the output image file, default:newOutput',default='newOutput')
parser.add_argument('--trainingInterval',help='Range of the files taken in the Training data folder to use for training, default: 0 10', type=int,nargs=2,default=[0 ,10])
parser.add_argument('--commonDataPath',help='Path of the folder containing trainingData, see ABIDEData 100 for example of internal structure, this argument is required')
parser.add_argument('--generateDistanceInfo',help='Test feature, default:0',type=int,default='0')
parser.add_argument('--ignoreLabelsNotInGenerated',help='In the generated images do not show labels not present in generated image, default:0',type=int,default='0')
listArg=parser.parse_args()
[start,end]=listArg.trainingInterval
commonPath=listArg.commonDataPath
outputPath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'Outputs',listArg.outputFile)
if os.path.isdir(outputPath): #uT.generateAllSlices expects a non-existing folder path
        shutil.rmtree(outputPath, ignore_errors=False, onerror=None)
print(start,end)
os.mkdir(outputPath)
f=open(os.path.join(outputPath,"outputInfo.txt"),"w+")

patientName=listArg.testFile


allKeyPaths=ut.getListFileKey(commonPath)
allAsegPaths=ut.getAsegPaths(commonPath)
allVolumePaths=ut.getBrainPath(commonPath)

trainingAsegPaths=allAsegPaths[start:end]
trainingVolumePaths=allVolumePaths[start:end]

testVolume=kt.getNiiData([x for x in allVolumePaths if patientName in x][0])
keyTest=ut.getKeypointFromOneFile([x for x in allKeyPaths if patientName in x][0])
truth=kt.getNiiData([x for x in allAsegPaths if patientName in x][0])


allKey=[]
for i in range(start,end):
    allKey.append(ut.getKeypointFromOneFile(allKeyPaths[i]))
y=0
for i in range(len(trainingAsegPaths)):
    if patientName in trainingAsegPaths[i-y]:
        allKey.pop(i)
        trainingAsegPaths.pop(i)
        trainingVolumePaths.pop(i)
        y=1
keyTrainingData=allKey

#do the segmentation
allMatches=kt.keypointDescriptorMatch(keyTest,keyTrainingData)
listMatches=kt.matchDistanceSelection(allMatches,keyTest,keyTrainingData)
listLabels=kt.getAllLabels(trainingAsegPaths,listMatches,keyTrainingData)
pMap,mLL=kt.voting(keyTest,keyTrainingData,listMatches,listLabels)
segMap,lMap,tabOfKeyTransfered=kt.doSeg(keyTest,listMatches,mLL,keyTrainingData,trainingAsegPaths,trainingVolumePaths,testVolume,pMap,listLabels,f,listArg.generateDistanceInfo)

#evaluate results

uTruth=np.unique(truth)
result=np.zeros((uTruth.shape[0],4))
result[:,0]=uTruth
print('\ndice coefficients')
f=open(os.path.join(outputPath,"outputInfo.txt"),"w+")
f.write(str(date.today()))
for j in range(uTruth.shape[0]):
    result[j,1]=np.sum(truth==uTruth[j])
    result[j,2]=np.sum(segMap==uTruth[j])
    if result[j,2]>0:
        result[j,3]=ut.getDC(segMap,truth,uTruth[j])
        f.write('\nlabel '+str(uTruth[j])+'\tdc:'+str(result[j,3]))
        print('label ',uTruth[j],'\tdc:',result[j,3])
        
ut.generateAllSlices(truth,segMap,outputPath,tabOfKeyTransfered,listArg.ignoreLabelsNotInGenerated)
f.close()