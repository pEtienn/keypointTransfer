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




kIP=kt.keyInformationPosition()

def CreateMaskKeyFiles(maskP,keyTestP,keyMaskP):
    maskF=os.listdir(maskP)
    keyTestF=os.listdir(keyTestP)
    for f in maskF:
        s1='_masked_gfc_reg.hdr'
        if s1 in f:
            n=f[:9]
            print(n)
            keyTestFP=os.path.join(keyTestP,next(x for x in keyTestF if n in x))
            maskFP=os.path.join(maskP,f)
            keyMaskFP=os.path.join(keyMaskP,f[:-3]+'key')
            mat=ut.getKeypointFromOneFile(keyTestFP)
            fW = open(keyMaskFP,"w", newline="\n")
            fR = open(keyTestFP,"r")
            for i in range(6):
                fW.write(fR.readline())
            
            img=nib.load(maskFP)
            arr=np.squeeze(img.get_fdata())>0
            for i in range(mat.shape[0]):
                if arr[tuple(np.int32(mat[i,kIP.XYZ]))]==True:
                    fW.write(fR.readline())
    
    fW.close()
    fR.close()



testP=str(Path(r"S:\skullStripData\test"))
maskP=str(Path(r"S:\skullStripData\mask"))
keyMaskP=str(Path(r"S:\skullStripData\keyMask"))
keyTestP=str(Path(r"S:\skullStripData\keyTest"))
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
keyTest=ut.getKeypointFromOneFile([x for x in allKeyTestPaths if patientName in x][0])
truth=ut.getKeypointFromOneFile([x for x in allKeyMaskPaths if patientName in x][0])


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

order=np.argsort(varPerMatches)
maskToKeep=5 #TO IMPROVE
dXYZ=np.int32(dXYZ)

firstMask=kt.getNiiData(maskPaths[order[0]])>0
marginXYZ=np.int32(2*np.amax(np.abs(dXYZ),axis=0))
maskCombined=np.zeros((firstMask.shape[0]+2*marginXYZ[0],firstMask.shape[1]+2*marginXYZ[1],firstMask.shape[2]+2*marginXYZ[2]))
tempMask=np.zeros((firstMask.shape[0]+2*marginXYZ[0],firstMask.shape[1]+2*marginXYZ[1],firstMask.shape[2]+2*marginXYZ[2]))
maskCombined[marginXYZ[0]+dXYZ[order[0],0]:marginXYZ[0]+dXYZ[order[0],0]+firstMask.shape[0],
             marginXYZ[1]+dXYZ[order[0],1]:marginXYZ[1]+dXYZ[order[0],1]+firstMask.shape[1],
             marginXYZ[2]+dXYZ[order[0],2]:marginXYZ[2]+dXYZ[order[0],2]+firstMask.shape[2]]+=firstMask
for i in range(1,maskToKeep):
    tempMask=np.zeros((firstMask.shape[0]+2*marginXYZ,firstMask.shape[1]+2*marginXYZ,firstMask.shape[2]+2*marginXYZ[2]))
    tempMask[marginXYZ[0]+dXYZ[order[i],0]:marginXYZ[0]+dXYZ[order[i],0]+firstMask.shape[0],
             marginXYZ[1]+dXYZ[order[i],1]:marginXYZ[1]+dXYZ[order[i],1]+firstMask.shape[1],
             marginXYZ[2]+dXYZ[order[i],2]:marginXYZ[2]+dXYZ[order[i],2]+firstMask.shape[2]]=kt.getNiiData(maskPaths[order[i]])>0
    maskCombined=np.logical_and(maskCombined,tempMask)
    
finalMask=maskCombined[marginXYZ[0]:marginXYZ[0]+firstMask.shape[0],
             marginXYZ[1]:marginXYZ[1]+firstMask.shape[1],
             marginXYZ[2]:marginXYZ[2]+firstMask.shape[2]]



keyTestFP=[x for x in allKeyTestPaths if patientName in x][0]
keyMaskFP=resultP
mat=ut.getKeypointFromOneFile(keyTestFP)
fW = open(keyMaskFP,"w", newline="\n")
fR = open(keyTestFP,"r")
for i in range(6):
    fW.write(fR.readline())
arr=finalMask
for i in range(mat.shape[0]):
    if arr[tuple(np.int32(mat[i,kIP.XYZ]))]==True:
        fW.write(fR.readline())

fW.close()
fR.close()