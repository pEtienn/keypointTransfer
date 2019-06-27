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

parser = argparse.ArgumentParser(description='Execute the keypoint transfer segmentation algorithm')
parser.add_argument('--testFile',help='Name of the test file, default:Caltech_0051456',default='Caltech_0051456')
parser.add_argument('--outputFile',help='Name of the output image file, default:newOutput',default='newOutput')
parser.add_argument('--trainingInterval',help='Range of the files taken in the ABIDEdata folder to use for training, default: 0 10', type=int,nargs=2,default=[0 ,10])
listArg=parser.parse_args()
[start,end]=listArg.trainingInterval
commonPath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'ABIDEdata')
outputPath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'Outputs',listArg.outputFile)
if os.path.isdir(outputPath):
    print('Output folder already exists, aborting program')
    exit()
    
print(start,end)
#print(listArg.outputFile)
#print(start,end)
#print(listArg.trainingPath)

patientName=listArg.testFile


allKeyfiles=ut.getListFileKey(commonPath)
allAsegPaths=ut.getAsegPaths(commonPath)
allBrainPaths=ut.getBrainPath(commonPath)

trainingAsegPaths=allAsegPaths[start:end]
trainingBrainPaths=allBrainPaths[start:end]

testBrain=kt.getNiiData(os.path.join(commonPath,'mri','mri_'+patientName+'.nii'))
testImage=ut.getKeypointFromOneFile(os.path.join(commonPath,'keypoint','key_'+patientName+'.key'))
truth=kt.getNiiData(os.path.join(commonPath,'segmentation','aseg_'+patientName+'.nii'))


allKey=[]
for i in range(start,end):
    allKey.append(ut.getKeypointFromOneFile(allKeyfiles[i]))
y=0
for i in range(len(trainingAsegPaths)):
    if patientName in trainingAsegPaths[i-y]:
        allKey.pop(i)
        trainingAsegPaths.pop(i)
        trainingBrainPaths.pop(i)
        y=1
trainingImages=allKey

#do the segmentation
allMatches=kt.keypointDescriptorMatch(testImage,trainingImages)
listMatches=kt.matchDistanceSelection(allMatches,testImage,trainingImages)
listLabels=kt.getAllLabels(trainingAsegPaths,listMatches,trainingImages)
pMap,mLL=kt.voting(testImage,trainingImages,listMatches,listLabels)
segMap,lMap=kt.doSeg(testImage,listMatches,mLL,trainingImages,trainingAsegPaths,trainingBrainPaths,testBrain,pMap,listLabels)

#evaluate results

uTruth=np.unique(truth)
result=np.zeros((uTruth.shape[0],4))
result[:,0]=uTruth
print('\ndice coefficients')
for j in range(uTruth.shape[0]):
    result[j,1]=np.sum(truth==uTruth[j])
    result[j,2]=np.sum(segMap==uTruth[j])
    if result[j,2]>0:
        result[j,3]=ut.getDC(segMap,truth,uTruth[j])
        print('label ',uTruth[j],'\tdc:',result[j,3])
        
ut.generateAllSlices(truth,segMap,outputPath,0)
