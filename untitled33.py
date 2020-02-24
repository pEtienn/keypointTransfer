# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:36:20 2020

@author: Etenne Pepyn
"""


import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location("module.KSlibrary", r'S:\siftTransfer\keypointTransfer\KSlibrary.py')
ks=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)

keysPath=r'S:\DataTestAlgo\fullHCP_hcpTrainingSEt\secondHalfOriginalHCP'
resultPath=r'S:\DataTestAlgo\fullHCP_hcpTrainingSEt\result'
brainSet=r"S:\DataTestAlgo\fullHCP_hcpTrainingSEt\brain_0_5.des"
skullSet=r"S:\DataTestAlgo\fullHCP_hcpTrainingSEt\skull_0_5.des"
groundTruthPath=r'S:\HCP_NoSkullStrip_T1w\trainingSet'
keySample=r"S:\HCP_NoSkullStrip_T1w\trainingSet\101410.key"
k=ks.ReadKeypoints(keySample)
pK=np.random.rand(k.shape[0],2)

ks._NormalizeDirectionalyX(pK,k)

ks.SkullStrip(keysPath,resultPath,brainSet,skullSet,printNonBrain=True,numberOfTrainingImages=10,normalize='Gaussian')
ks.EvaluateSkullStrip(resultPath,groundTruthPath,nameRegex='^(\d+)')