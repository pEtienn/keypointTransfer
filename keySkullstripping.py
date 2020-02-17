# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:12:15 2019

import sys
sys.argv=['',r'S:\testMatch\OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc\orginalNotMatchedWithSS.key']
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

# keyTestP=str(Path(r"S:\originalOASISdata\OASIS_head\key"))
keyTestP=str(Path(r"S:\skullStripData\HCP_Twin_Sample\all\key"))
# resultP=str(Path(r"S:\keySkullStripping\results"))
resultP=str(Path(r"S:\skullStripData\HCP_Twin_Sample\all\keyFilteredAlgo"))
trainingSetPath=str(Path(r'S:\skullStripData\trainingSet'))#ConsiderScale
allKeyTestPaths=ut.listdir_fullpath(keyTestP)
numberDetectionRegex='([0-9]+).'
# numberDetectionRegex='OAS._([0-9]{4})_'  #detect the number in path following this format:
#S:\skullStripData\keyMaskMany\OAS1_0002_MR1_mpr_nn_anon_111_t88_masked_gfc_reg.key
#use of regex will return '0002' on that path
rPatientNumber=re.compile(numberDetectionRegex)
listTestNumber=[]
allTestPath=allKeyTestPaths[0:4]
for testPath in allTestPath:
    listTestNumber.append(rPatientNumber.findall(testPath)[0])

[brainFilePaths, skullFilePaths]=ks.GenerateFilePaths(trainingSetPath)
[brainFilePaths, skullFilePaths]=ks.RemoveTestPatientFromFilePaths(listTestNumber,brainFilePaths,skullFilePaths)
    
allKey=ks.CombineAllKey(brainFilePaths,skullFilePaths)
[listFlann,listParam]=ks.GenerateSearchTree(allKey)

for i in range(len(allTestPath)):
    testPath=allTestPath[i]
    testNumber=listTestNumber[i]
    testKey=ks.ReadKeypoints(testPath)
    [resolution,header]=ks.GetResolutionHeaderFromKeyFile(testPath)
    # trueBrainPath=os.path.join(trainingSetPath,'brainKey',[x for x in os.listdir(os.path.join(trainingSetPath,'brainKey')) if testNumber in x][0])
    # keyTrueBrain=ks.ReadKeypoints(trueBrainPath)
    

    [keyBrain,keySkull]=ks._SkullStrip(testKey,listFlann,listParam,len(brainFilePaths),resolution,normalizeProbabilitySpatially=True)
                                      # ,patientRepertory=allPatient,keyTrueBrain=keyTrueBrain,patientName=testNumber,doRandomTest=False)
    
    if True:
        ks.WriteKeyFile(os.path.join(resultP,testNumber+'BrainNormalized.key'),keyBrain,header=header)
        ks.WriteKeyFile(os.path.join(resultP,testNumber+'SkullNormalized.key'),keySkull,header=header)

# print('avg key brain DC: ',allPatient.GetAvgKeyBrainDC())
# print('avg key skull DC: ',allPatient.GetAvgKeySkullDC())

        
    