from rdkit import Chem
from rdkit.Chem import rdMolAlign
from .torsions import DihedralsInfo, FromTorsionLib
from collections import defaultdict

def SortEnsembleByTABS(m):
    """
    Sorts the conformers of a molecule into groups based on their Torsion Angular Bin Strings (TABS).

    This function processes a molecule with multiple conformers, extracts the TABS for each conformer,
    and groups the conformers by their TABS values. The result is a dictionary where the keys are
    TABS strings and the values are lists of conformer indices that share the same TABS.

    :param m: The molecule object to process. It must have conformers and support the TorsionLib
              functionality for extracting dihedral information.
    :raises ValueError: If the molecule does not contain any conformers.
    :return: A dictionary mapping TABS strings to lists of conformer indices.
    """
    if m.GetNumConformers() < 1:
        raise ValueError("No conformers found in molecule.")
    info = FromTorsionLib(m)
    allTabs = info.GetTABS()
    sortedByTabs = defaultdict(list)
    for i, t in enumerate(allTabs):
        sortedByTabs[t].append(i)
    return sortedByTabs

def AnalyzeTABSforIntraRmsd(m, sortedTabsDict):
    """
    :param m: The molecule to analyze.
    :param sortedTabsDict: A dictionary containing the sorted TABS.
    :return: A dictionary containing the calculated RMSDs for each key in the sortedTabsDict.
    """
    rmsds = {}
    # copying over only for debugging reasons
    molCopy = Chem.Mol(m)
    molCopy = Chem.RemoveHs(molCopy)
    for key in sortedTabsDict:
        tmp = []
        idxs = sortedTabsDict[key]
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                tmp.append(rdMolAlign.GetBestRMS(molCopy, molCopy, idxs[i], idxs[j]))
        rmsds[key] = tmp
    return rmsds

def AnalyzeTABSforInterRmsd(m, sortedTabsDict):
    """
    :param m: The input molecule to analyze.
    :param sortedTabsDict: A dictionary containing sorted TABS data.
    :returns: A dictionary containing the inter-RMSD values for each key in sortedTabsDict.
    """
    rmsds = {}
    # copying over only for debugging reasons
    molCopy = Chem.Mol(m)
    molCopy = Chem.RemoveHs(molCopy)
    for k, key in enumerate(sortedTabsDict):
        tmp = []
        idx = sortedTabsDict[key]
        for k2, key2 in enumerate(sortedTabsDict):
            if k2 != k:
                idx2 = sortedTabsDict[key2]
                for id in idx:
                    for id2 in idx2:
                        tmp.append(rdMolAlign.GetBestRMS(molCopy, molCopy, id, id2))
        rmsds[key] = tmp
    return rmsds