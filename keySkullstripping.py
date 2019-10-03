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
            keyTestFP=os.path.join(keyTestP,next(x for x in keyTestF if '0001' in x))
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


fullP=str(Path(r"S:\OASISkey\OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.key"))
testP=str(Path(r"S:\skullStripData\test"))
maskP=str(Path(r"S:\skullStripData\mask"))
keyMaskP=str(Path(r"S:\skullStripData\keyMask"))
keyTestP=str(Path(r"S:\skullStripData\keyTest"))
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
    allKey.append(ut.getKeypointFromOneFile(allKeyMaskPaths[i]))
y=0
for i in range(len(allMaskPaths)):
    if patientName in allMaskPaths[i-y]:
        allKey.pop(i)
        allMaskPaths.pop(i)
        allKeyMaskPaths.pop(i)
        y=1
keyTrainingData=allKey
allMatches=kt.keypointDescriptorMatch(keyTest,keyTrainingData)