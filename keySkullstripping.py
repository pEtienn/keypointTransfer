# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:12:15 2019

import sys
sys.argv=['',r'S:\keySkullStripping\results\kBrain.key']
execfile(r'S:\keySkullStripping\Python\visualizeFeatures.py')
@author: Etenne Pepyn
"""
import os
from pathlib import Path
import numpy as np 
import nibabel as nib
import re
import utilities as ut
import KSlibrary as ks

allPatient=ks.PatientRepertory()

keyTestP=str(Path(r"S:\skullStripData\keyTestFew"))
resultP=str(Path(r"S:\keySkullStripping\results"))
trainingSetPath=str(Path(r'S:\skullStripData\trainingSet'))
allKeyTestPaths=ut.listdir_fullpath(keyTestP)
numberDetectionRegex='OAS._([0-9]{4})_'  #detect the number in path following this format:
#S:\skullStripData\keyMaskMany\OAS1_0002_MR1_mpr_nn_anon_111_t88_masked_gfc_reg.key
#use of regex will return '0002' on that path
rPatientNumber=re.compile(numberDetectionRegex)

for testPath in allKeyTestPaths:
    testNumber=rPatientNumber.findall(testPath)[0]
    [testKey,resolution,header]=ks.GetKeypointsResolutionHeader(testPath)
    trueBrainPath=os.path.join(trainingSetPath,'brainKey',[x for x in os.listdir(os.path.join(trainingSetPath,'brainKey')) if testNumber in x][0])
    [keyTrueBrain,r,h]=ks.GetKeypointsResolutionHeader(trueBrainPath)
    
    [brainFilePaths, skullFilePaths]=ks.GenerateFilePaths(trainingSetPath)
    [brainFilePaths, skullFilePaths]=ks.RemoveTestPatientFromFilePaths(testNumber,brainFilePaths,skullFilePaths)
    
    allKey=ks.CombineAllKey(brainFilePaths,skullFilePaths)

    [keyBrain,keySkull]=ks.SkullStrip(testKey,allKey,len(brainFilePaths),resolution,normalizeProbabilitySpatially=False,patientRepertory=allPatient,
    keyTrueBrain=keyTrueBrain,patientName=testNumber,doRandomTest=True)
    
    if False:
        ks.WriteKeyFile(os.path.join(resultP,testNumber+'BrainNormalized.key'),keyBrain,header=header)
        ks.WriteKeyFile(os.path.join(resultP,testNumber+'SkullNormalized.key'),keySkull,header=header)

print('avg key brain DC: ',allPatient.GetAvgKeyBrainDC())
print('avg key skull DC: ',allPatient.GetAvgKeySkullDC())

        
    