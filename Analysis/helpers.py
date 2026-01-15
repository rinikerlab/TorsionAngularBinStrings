import numpy as np
from tabs import custom
from sklearn.metrics import confusion_matrix

def GetTabsPopulationMatrix(uniqueTabs, counts):
    """
    Aggregate counts into a 2D population matrix based on two-state tabs identifiers.

    The function expects each element of `uniqueTabs` to be convertible to a string whose
    first two characters are decimal digits representing two 1-based state indices
    (for example "12" means state A=1, state B=2). For each tabs identifier and the
    corresponding count in `counts`, the function increments the cell at
    (row = stateA-1, col = stateB-1) in the returned NumPy integer array.

    Parameters
    ----------
    uniqueTabs : Iterable
        Iterable of tabs identifiers.
    counts : Iterable of int
        Iterable of non-negative integer counts with the same length as
        `uniqueTabs`.

    Returns
    -------
    numpy.ndarray
        2D integer array of shape (maxA, maxB) where `maxA` is the maximum A-state
        value found and `maxB` is the maximum B-state value found. Cells contain
        the summed counts for each (A,B) pair. Indices in the array are zero-based
        (state 1 maps to index 0).

    Notes
    -----
    - The function interprets state indices as 1-based in the input and converts
      them to 0-based indices for the output matrix.
    - Only the first two characters of each tabs identifier are used. If state
      numbers can be multi-digit, pre-process identifiers to a consistent format
      (e.g., use a delimiter or a tuple of two integers) before calling this function.

    Examples
    --------
    >>> uniqueTabs = ["11", "12", "21"]
    >>> counts = [5, 3, 2]
    >>> GetTabsPopulationMatrix(uniqueTabs, counts)
    array([[5, 3],
           [2, 0]])
    """
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
    """
    Determine which dihedral candidates are relevant for correlation analysis.

    This function evaluates a list of dihedral candidates and returns those
    considered "relevant" based on the distribution of TABS-assigned states
    obtained from the provided per-dihedral profiles. A candidate is marked
    relevant only if it has more than one observed state and the smallest
    state-population percentage across its observed states is greater than
    the provided threshold.

    Parameters
    ----------
    mol : object
        Molecular object passed to custom.CustomDihedralInfo. 
    candidates : iterable
        Iterable of dihedral definitions.
    profiles : dict
        Mapping from tuple(dihedral) -> profile object. Each profile is passed
        as `confTorsions` to custom.CustomDihedralInfo and used to compute
        TABS population assignments.
    threshold : float, optional
        Percentage threshold (in percent, 0-100) used to decide relevance.
        Default is 1.0. A candidate is considered relevant only if the
        smallest state population percentage for that candidate is strictly
        greater than `threshold`.

    Returns
    -------
    list
        List of dihedral candidates (in the same form as provided in
        `candidates`) that passed the relevance criteria.
    """
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
                if len(unique) < 3:
                    print(f"Dihedral {dihedral} not relevant: {percentages}")
                else:
                    relevant.append(dihedral)
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
    """
    Compute positive and negative predictive values from a confusion matrix.

    Parameters
    ----------
    cm : dict
        A dictionary expected to contain a 2x2 confusion matrix under the key 'cm'.
        The confusion matrix must be convertible to a flat sequence of four elements
        in the order [tn, fp, fn, tp] when flattened/raveled (e.g., a 2x2 numpy array).
    debug : bool, optional
        If True, print intermediate values used to compute precision (tp and fp).
        Default is False.

    Returns
    -------
    tuple of (float, float)
        A tuple (ppv, npv) where:
        - ppv (positive predictive value / precision) = tp / (tp + fp), or 0.0 if (tp + fp) == 0
        - npv (negative predictive value) = tn / (fn + tn), or 0.0 if (fn + tn) == 0

    Examples
    --------
    Assuming a numpy 2x2 confusion matrix:
    >>> cm = {'cm': np.array([[50, 10],
    ...                       [ 5, 35]])}
    >>> AnalyticsOnConfusionMatrices(cm)
    (0.7777777777777778, 0.9090909090909091)
    """
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

def Shift(x):
    # for array x check for every single entry if larger than 180, then subtract from 360
    return np.array([angle-360 if angle>180 else angle for angle in x])