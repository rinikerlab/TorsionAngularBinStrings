import numpy as np
from tabs import custom

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