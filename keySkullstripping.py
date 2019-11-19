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
import keypointTransfer as kt
import KSlibrary as ks
from pyflann import *

#***********************PATHS AND KEYS
kIP=kt.keyInformationPosition()
allPatient=ks.PatientRepertory()

testP=str(Path(r"S:\skullStripData\test"))
maskP=str(Path(r"S:\skullStripData\mask"))
keyMaskP=str(Path(r"S:\skullStripData\keyMaskFew"))
keyTestP=str(Path(r"S:\skullStripData\keyTestFew"))
resultP=str(Path(r"S:\keySkullStripping\results\r1.key"))
allTestPaths=ut.listdir_fullpath(testP)
allMaskPaths=ut.listdir_fullpath(maskP)
allKeyTestPaths=ut.listdir_fullpath(keyTestP)
allKeyMaskPaths=ut.listdir_fullpath(keyMaskP)
numberDetectionRegex='OAS._([0-9]{4})_'  #detect the number in path following this format:
#S:\skullStripData\keyMaskMany\OAS1_0002_MR1_mpr_nn_anon_111_t88_masked_gfc_reg.key
#use of regex will return '0002' on that path
rPatientNumber=re.compile(numberDetectionRegex)
patientNames=[]
[start,end]=[0,len(allKeyTestPaths)-1]
for i in allKeyTestPaths:
    patientNames.append(rPatientNumber.findall(i)[0])

for patientName in [patientNames[0]]:#patientNames:
    
    #FILE SELECTION
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

    
    maskPaths=allMaskPaths[start:end]
    keyMaskPaths=allKeyMaskPaths[start:end]
    keyPaths=allKeyTestPaths[start:end]
    
    testVolume=kt.getNiiData([x for x in allTestPaths if patientName in x][0])
    truth=ut.getKeypointFromOneFile([x for x in allKeyMaskPaths if patientName in x][0])
    keyTestPath=[x for x in allKeyTestPaths if patientName in x][0]
    [keyTest,resolutionTest,h]=ks.GetKeypointsResolutionHeader(keyTestPath)
    
    brainShape=testVolume.shape #to change
    listTrainingKey=[]
    for i in range(start,end):
        listTrainingKey.append(ut.getKeypointFromOneFile(keyPaths[i]))
    y=0
    for i in range(len(maskPaths)):
        if patientName in maskPaths[i-y]:
            listTrainingKey.pop(i)
            maskPaths.pop(i)
            keyMaskPaths.pop(i)
            keyPaths.pop(i)
            y=1
    keyTrainingData=listTrainingKey
    
    maskTrueBrain=kt.getNiiData([x for x in allMaskPaths if patientName in x][0])>0
    
    #Generate key data structure
    nbKeys=0
    keyIndexes=np.zeros((len(listTrainingKey),2))
    for i in range(len(listTrainingKey)):
        nbKey=listTrainingKey[i].shape[0]
        keyIndexes[i,0]=nbKeys
        nbKeys+=nbKey
        keyIndexes[i,1]=nbKeys
        
        
    allTrainingKey=np.zeros((nbKeys,83))
    indexCounter=0
    for i in range(len(listTrainingKey)):
        keyToAdd=listTrainingKey[i]
        brainKey=ut.getKeypointFromOneFile(keyMaskPaths[i])
        skullKey=ks.SubstractKeyImages(keyToAdd,brainKey)
        allTrainingKey[indexCounter:indexCounter+brainKey.shape[0],:-1]=np.concatenate((brainKey,np.ones((brainKey.shape[0],1))),axis=1)
        allTrainingKey[indexCounter:indexCounter+brainKey.shape[0],-1]=i
        indexCounter+=brainKey.shape[0]
        allTrainingKey[indexCounter:indexCounter+skullKey.shape[0],:-1]=np.concatenate((skullKey,np.zeros((skullKey.shape[0],1))),axis=1)
        allTrainingKey[indexCounter:indexCounter+skullKey.shape[0],-1]=i
        indexCounter+=skullKey.shape[0]
    

    #***************** NN
    viewBrain=allTrainingKey[allTrainingKey[:,81]==1,:]
    viewSkull=allTrainingKey[allTrainingKey[:,81]==0,:]
    flannB = FLANN()
    flannS= FLANN()
    paramsB = flannB.build_index(viewBrain[:,kIP.descriptor], algorithm="kdtree",trees=4);
    paramsS=flannS.build_index(viewSkull[:,kIP.descriptor], algorithm="kdtree",trees=4);
    pK=np.zeros((keyTest.shape[0],2))
    nbNN=len(listTrainingKey)
    
    for i in range(keyTest.shape[0]):
        resultB, distB = flannB.nn_index(keyTest[i,kIP.descriptor],nbNN, checks=paramsB["checks"])
        distAB=np.squeeze(np.asarray(distB))
        var=np.sqrt(distAB[0])
        exp=np.exp(-distAB/(2*np.power(var,2)))
        pK[i,0]=np.sum(exp)
        resultS, distS = flannS.nn_index(keyTest[i,kIP.descriptor],nbNN, checks=paramsS["checks"])
        distAS=np.squeeze(np.asarray(distS))
        pK[i,1]=np.sum(np.exp(-distAS/(2*np.power(var,2))))
    #*****************
#    pKSave=pK
#    pMap=ks.GenerateNormalizedProbabilityMap(np.power(pK,2),keyTest,brainShape)
#    pK=ks.GetProbabilityKeyFromPMap(pMap,keyTest)
    
#    mask=pMap[:,:,:,0]>pMap[:,:,:,1]
    
#    keyBrain=ks.FilterKeysWithMask(keyTest,mask)
#    keySkull=ks.FilterKeysWithMask(keyTest,~mask)
    #TEMP Classification
    indexBrain=pK[:,0]>pK[:,1]
#    indexBrain=ks.ClassifyByRatio(pK)
    
    #random tests *************
#    r=np.random.rand((pK.shape[0]))
#    indexBrain=(r<=0.95)
    keyBrain=keyTest[indexBrain]
    keySkull=keyTest[~indexBrain]
    patientObject=ks.Patient(keyTest,truth,keyBrain=keyBrain,keySkull=keySkull)
    allPatient.AddPatient(patientName,patientObject)
    patientObject.PrintStats()
        
print('avg key brain DC: ',allPatient.GetAvgKeyBrainDC())
print('avg key skull DC: ',allPatient.GetAvgKeySkullDC())
    #******************BEST MATCH SELECTION
#    allMatches=kt.keypointDescriptorMatch(keyTest,keyTrainingData)
#    nbTrainingData=len(keyTrainingData)
#    listMatches=kt.matchDistanceSelection(allMatches,keyTest,keyTrainingData)
    
    
#    [keyTrueBrain,brainMask,skullMask]=ks.generateMask(listMatches,keyTrainingData,keyTest,resolutionTest,kIP,maskPaths,allKeyMaskPaths,patientName)
#    
#    
#    patientObject=ks.Patient(keyTest,keyTrueBrain,brainMask,skullMask,maskTrueBrain)
#    allPatient.addPatient(patientName,patientObject)
#    patientObject.printStats()

        
    