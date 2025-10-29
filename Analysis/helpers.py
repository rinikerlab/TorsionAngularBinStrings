import numpy as np
from tabs import custom
from sklearn.metrics import confusion_matrix

def GetTabsPopulationMatrix(uniqueTabs, counts):
    statesA = set()
    statesB = set()
    for tabs in uniqueTabs:
        s = str(tabs)
        statesA.add(int(s[0]))
        statesB.add(int(s[1]))
    # get the maximum in statesA and statesB
    maxA = max(statesA)
    maxB = max(statesB)
    # create a matrix with the size of maxA x maxB
    matrix = np.zeros((maxA, maxB), dtype=int)
    for tabs, count in zip(uniqueTabs, counts):
        s = str(tabs)
        a = int(s[0]) - 1  # convert to zero-based index
        b = int(s[1]) - 1  # convert to zero-based index
        matrix[a, b] += count
    return matrix

def CheckForCorrelationCandidates(mol, candidates, profiles, threshold=1):
    relevant = []
    # not relevant for correlation analysis if there is only one state
    for dihedral in candidates:
        customProfile = profiles[tuple(dihedral)]
        info = custom.CustomDihedralInfo(mol, [dihedral], customProfile, showFits=False)
        tabsPopulationTraj = info.GetTABS(confTorsions=customProfile)
        unique, counts = np.unique(tabsPopulationTraj, return_counts=True)
        percentages = counts / len(tabsPopulationTraj) * 100
        if len(unique) > 1:
            # check if the percentage of the most populated state is above the threshold
            if min(percentages) > threshold:
                relevant.append(dihedral)
            else:
                print(f"Dihedral {dihedral} not relevant: {percentages}")
        else:
            print(f"Dihedral {dihedral} not relevant: {percentages}")
    return relevant

def GetMetrics(yTrue, yPred, comparisonMeasureCutoffTrue, comparisonMeasureCutoffPred, comparisonid):
    """
    Compute binary classification metrics by converting continuous/score inputs
    into binary labels according to a selected comparison mode, then computing
    a confusion matrix and derived predictive values.

    Parameters
    ----------
    yTrue : array-like
        Ground-truth values. Depending on `comparisonid`, these may be treated
        as continuous scores (kept as-is) or thresholded to produce binary
        labels. Accepts lists, numpy arrays, or other sequence types.
    yPred : array-like
        Predicted values or scores. Depending on `comparisonid`, these may be
        treated as continuous scores (kept as-is) or thresholded to produce
        binary labels. Accepts lists, numpy arrays, or other sequence types.
    comparisonMeasureCutoffTrue : float
        Numeric threshold used to convert `yTrue` into binary labels when the
        selected comparison mode requires thresholding.
    comparisonMeasureCutoffPred : float
        Numeric threshold used to convert `yPred` into binary labels when the
        selected comparison mode requires thresholding.
    comparisonid : int
        Selector that determines how `yTrue` and `yPred` are converted to binary
        labels before metric calculation. Supported values:
          - 1: true is RMSD (thresholded as < comparisonMeasureCutoffTrue -> 1),
               pred is shape combined (thresholded as > comparisonMeasureCutoffPred -> 1)
          - 2: true is tabs (kept as-is), pred is shape combined (thresholded as > comparisonMeasureCutoffPred -> 1)
          - 3: true is RMSD (thresholded as < comparisonMeasureCutoffTrue -> 1), pred is tabs (kept as-is)
          - 4: true is shape (thresholded as > comparisonMeasureCutoffTrue -> 1), pred is tabs (kept as-is)
          - 5: true is shape (thresholded as > comparisonMeasureCutoffTrue -> 1), pred is RMSD (thresholded as < comparisonMeasureCutoffPred -> 1)

    Returns
    -------
    dict
        A dictionary containing:
          - "cm": ndarray, the 2x2 confusion matrix with label order [0, 1]
          - "ppv": float or array-like, positive predictive value(s) computed by
                   AnalyticsOnConfusionMatrices
          - "npv": float or array-like, negative predictive value(s) computed by
                   AnalyticsOnConfusionMatrices
    """
    # comparsionid 1: true is rmsd, pred is shape combined
    # comparsionid 2: true is tabs, pred is shape combined
    # comparsionid 3: true is rmsd, pred is tabs
    # comparisonid 4: true is shape, pred is tabs
    # comparisonid 5: true is shape, pred is rmsd
    metrics = {}
    if comparisonid == 1:
        yPred = (np.array(yPred) > comparisonMeasureCutoffPred )*1
        yTrue = (np.array(yTrue) < comparisonMeasureCutoffTrue )*1
    elif comparisonid == 2:
        yPred = (np.array(yPred) > comparisonMeasureCutoffPred )*1
        yTrue = np.array(yTrue)
    elif comparisonid == 3:
        yPred = np.array(yPred)
        yTrue = (np.array(yTrue) < comparisonMeasureCutoffTrue )*1
    elif comparisonid == 4:
        yPred = np.array(yPred)
        yTrue = (np.array(yTrue) > comparisonMeasureCutoffTrue )*1
    elif comparisonid == 5:
        yPred = (np.array(yPred) < comparisonMeasureCutoffPred )*1
        yTrue = (np.array(yTrue) > comparisonMeasureCutoffTrue )*1
    cm = confusion_matrix(yTrue,yPred,labels=[0,1])
    if cm.size == 0:
        print("Empty confusion matrix")
    metrics["cm"] = cm
    ppv, npv = AnalyticsOnConfusionMatrices(metrics)
    metrics['ppv'] = ppv
    metrics['npv'] = npv
    return metrics

def AnalyticsOnConfusionMatrices(cm, debug=False):
    tn, fp, fn, tp = cm['cm'].ravel()
    if (tp + fp) == 0:
        ppv = 0.0
    else:
        ppv = (tp/(tp+fp)) # precision
        if debug:
            print(f"tp: {tp}, fp: {fp}")
            print(tp+fp)
            print("***")
    if (fn + tn) == 0:
        npv = 0.0
    else:
        npv = (tn/(fn+tn))
    return ppv, npv