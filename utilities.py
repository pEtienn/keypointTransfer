import nibabel as nib
import os

def getDataFromOneFile(filePath):
    
    file= open (filePath,'r')
    #skip
    for i in range(6):
        temp=file.readline()    
    end=0
     
    
    lineString=file.readline()
    fileData=np.fromstring(lineString, dtype=float,sep='\t')
    
    while end==0:
        lineString=file.readline()
        if lineString!="":
            floatLine=np.fromstring(lineString, dtype=float,sep='\t')
            fileData=np.vstack((fileData,floatLine))
        else:
            end=1
    file.close
    return fileData

def getListFileKey(commonKeyPath):
    allFolders=os.listdir(commonKeyPath)
    allKeyFiles=[]
    for file in allFolders:
        if os.path.isdir(commonKeyPath+file):
            proposedPath=commonKeyPath+file+'/'+'mri/brain.key'
            if os.path.isfile(proposedPath):
                allKeyFiles.append(proposedPath)
    return allKeyFiles

def getAsegPaths(allKeyFiles):
    allNiiPaths=[]
    for i in range(len(allKeyFiles)):
        s=allKeyFiles[i].replace('preprocessed','ABIDE_aseg')
        s=s.replace('brain.key','aseg.nii')
        if os.path.isfile(s):
            allNiiPaths.append(s)
        else:
            print("missing file:", s)
    return allNiiPaths

def getBrainPath(allKeyFiles):
    allBrainPaths=[]
    for i in range(len(allKeyFiles)):
        s=allKeyFiles[i]
        s=s.replace('brain.key','brain.nii')
        if os.path.isfile(s):
            allBrainPaths.append(s)
        else:
            print("missing file:", s)
    return allBrainPaths

def getNiiData(niiPath):
    img=nib.load(niiPath)
    return img.get_fdata()
