# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:12:15 2019

@author: Etenne Pepyn
"""
import os
from pathlib import Path
import numpy as np 
import nibabel as nib
import re
import utilities as ut
import keypointTransfer as kt
import KSlibrary as ks

#***********************PATHS AND KEYS
kIP=kt.keyInformationPosition()

testP=str(Path(r"S:\skullStripData\test"))
maskP=str(Path(r"S:\skullStripData\mask"))
keyMaskP=str(Path(r"S:\skullStripData\keyMaskMany"))
keyTestP=str(Path(r"S:\skullStripData\keyTestMany"))
resultP=str(Path(r"S:\keySkullStripping\results\r1.key"))
[start,end]=[0,15]
patientName='0001'
allTestPaths=ut.listdir_fullpath(testP)
allMaskPaths=ut.listdir_fullpath(maskP)

y=0
z=0
ty=allTestPaths
tz=allMaskPaths
for i in range(len(allTestPaths)):
    if ty[i-y][-3:]=='img':
        allTestPaths.pop(i-y)
        y+=1
    if tz[i-z][-3:]=='img':
        allMaskPaths.pop(i-z)
        z+=1
allKeyTestPaths=ut.listdir_fullpath(keyTestP)
allKeyMaskPaths=ut.listdir_fullpath(keyMaskP)

maskPaths=allMaskPaths[start:end]
keyMaskPaths=allKeyMaskPaths[start:end]

testVolume=kt.getNiiData([x for x in allTestPaths if patientName in x][0])
truth=ut.getKeypointFromOneFile([x for x in allKeyMaskPaths if patientName in x][0])
keyTestPath=[x for x in allKeyTestPaths if patientName in x][0]
[keyTest,resolutionTest,h]=ks.GetKeypointsResolutionHeader(keyTestPath)

allKey=[]
for i in range(start,end):
    allKey.append(ut.getKeypointFromOneFile(keyMaskPaths[i]))
y=0
for i in range(len(maskPaths)):
    if patientName in maskPaths[i-y]:
        allKey.pop(i)
        maskPaths.pop(i)
        keyMaskPaths.pop(i)
        y=1
keyTrainingData=allKey

#******************BEST MATCH SELECTION
allMatches=kt.keypointDescriptorMatch(keyTest,keyTrainingData)
nbTrainingData=len(keyTrainingData)
listMatches=kt.matchDistanceSelection(allMatches,keyTest,keyTrainingData)
varPerMatches=np.zeros(len(listMatches))
dXYZ=np.zeros((len(listMatches),3))
for i in range(len(listMatches)):
   
    matches=listMatches[i]
    keyTrainingDatum=keyTrainingData[i]
    
    testXYZ=keyTest[np.int32(matches[:,0]),kIP.XYZ]
    trainingXYZ=keyTrainingDatum[np.int32(matches[:,1]),kIP.XYZ]
    matchedXYZDifference=testXYZ-trainingXYZ      
    varPerMatches[i]=np.var(matchedXYZDifference[:,0])+np.var(matchedXYZDifference[:,1])+np.var(matchedXYZDifference[:,2]) #TO IMPROVE
    [unused,dXYZ[i,:]]=kt.houghTransformGaussian(matchedXYZDifference)


brainMask=ks.CreateCombinedMask(maskPaths,varPerMatches,dXYZ,resolutionTest)
skullMask=ks.CreateCombinedMask(maskPaths,varPerMatches,dXYZ,resolutionTest,brainMask=False) # skull and background ==1

marginMask=~(brainMask+skullMask)
#**************Filtering Keypoints
keyBrain=ks.FilterKeysWithMask(keyTest,brainMask)
keyMargin=ks.FilterKeysWithMask(keyTest,marginMask)
keySkull=ks.FilterKeysWithMask(keyTest,skullMask)
[keyTrue,r,h]=ks.GetKeypointsResolutionHeader([x for x in allKeyMaskPaths if patientName in x][0])
keySkullTrue=ks.SubstractKeyImages(keyTest,keyTrue)
print('brain: ',keyBrain.shape[0])
print('margin: ',keyMargin.shape[0])
print('skull: ',keySkull.shape[0])
print('|true Brain|: ',keyTrue.shape[0])
print('|true positive|: ',ks.CompareKeyImages(keyBrain,keyTrue))
print('|false positive|: ',ks.CompareKeyImages(keyBrain,keySkullTrue))
print('|true negative|: ',ks.CompareKeyImages(keySkull,keySkullTrue))
print('|false negative|: ' ,ks.CompareKeyImages(keySkull,keyTrue))
print('brain in margin: ',ks.CompareKeyImages(keyMargin,keyTrue))
print('skull in margin: ',ks.CompareKeyImages(keyMargin,keySkullTrue))
trueMask=kt.getNiiData([x for x in allMaskPaths if patientName in x][0])>0
print('mask dc: ',ut.getDC(brainMask,trueMask,1))

class patientStat:
    
    def __init__(self, keyTrueBrain, keyTrueSkull, keySkull, keyMargin, keyBrain,maskBrain,maskSkull,maskTrueBrain):
        self.mBrain=maskBrain 
        self.mSkull=maskSkull
        self.mMargin=~(brainMask+skullMask)
        
        self.ktBrain=keyTrueBrain
        self.ktSkull=keyTrueSkull
        self.kSkull=keySkull
        self.kMargin=keyMargin
        self.kBrain=keyBrain
        
    