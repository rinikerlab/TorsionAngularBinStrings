import numpy as np
from sklearn.metrics import confusion_matrix
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, TorsionFingerprints
import pandas as pd

def GetRawData(filenames):
    allData = []
    for filename in filenames:
        try:
            rmsdTabsData = pd.read_csv(filename, header=None)
            col1 = rmsdTabsData[2].to_numpy()
            col2 = rmsdTabsData[3].to_numpy()
            tmp = np.concatenate((col1.reshape(-1, 1), col2.reshape(-1, 1)), axis=1)
            if not np.isnan(tmp).any():
                allData.append(tmp)
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue
    return allData

def GetMetrics(yTrue, yPred, comparisonMeasureCutoff):
    metrics = {}
    yTrue = (yTrue < comparisonMeasureCutoff )*1
    if comparisonMeasureCutoff == 0.0:
        assert np.sum(yTrue) == 0.0, "not only negatives"
    cm = confusion_matrix(yTrue,yPred,labels=[0,1])
    if cm.size == 0:
        print("Empty confusion matrix")
    metrics["cm"] = cm
    return metrics

def GetAllConfusionMatrices(rawData, rmsdThresholds):
    allAna = {}
    for i, data in enumerate(rawData):
        yPred = data[:,0]
        tmp = {}
        for rmsdThreshold in rmsdThresholds:
            yTrue = data[:,1]
            try: 
               tmp[np.round(rmsdThreshold,1)] = GetMetrics(yTrue,yPred,rmsdThreshold) 
            except:
                continue
        allAna[i] = tmp
    return allAna

def GetAllRmsdConfusionMatrix(anaDict, rmsd):
    allCms = []
    for key in anaDict.keys():
        try: 
            allCms.append(anaDict[key][np.round(rmsd,1)]['cm'])
        except:
            print(key)
            continue
    return np.sum(allCms,axis=0)

def AnalyticsOnConfusionMatrices(cm):
    positivePredictiveValues = []
    negativePredictiveValues = []
    for rmsdThres in cm.keys():
        tn, fp, fn, tp = cm[np.round(rmsdThres,1)].ravel()
        positivePredictiveValues.append(tp/(tp+fp)) # precision
        negativePredictiveValues.append(tn/(fn+tn))
    return positivePredictiveValues, negativePredictiveValues

def SumConfusionMatrices(cms,rmsdThresholds):
    summedCms = {}
    for rmsdThres in rmsdThresholds:
        summedCms[np.round(rmsdThres,1)] = GetAllRmsdConfusionMatrix(cms,rmsdThres)
    return summedCms

def ListOfRawDataToDifferentSameOneArrays(rawData):
    for i, data in enumerate(rawData):
        if i == 0:
            same = data[data[:,0] == 1.0]
            different = data[data[:,0] == 0.0]
        else:
            same = np.concatenate((same,data[data[:,0] == 1.0]),axis=0)
            different = np.concatenate((different,data[data[:,0] == 0.0]),axis=0)    
    return same, different