# Copyright (C) 2026 ETH Zurich, Jessica Braun, and other TABS contributors.
# All rights reserved.
# This file is part of TABS.
# The contents are covered by the terms of the MIT license
# which is included in the file LICENSE.

from rdkit import Chem
from rdkit.Chem import rdMolAlign
from .torsions import DihedralInfoFromTorsionLib
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
    info = DihedralInfoFromTorsionLib(m)
    allTabs = info.GetTABS()
    sortedByTabs = defaultdict(list)
    for i, t in enumerate(allTabs):
        sortedByTabs[t].append(i)
    return sortedByTabs

def AnalyzeTABSForIntraRMSD(m, sortedTabsDict):
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

def AnalyzeTABSForInterRMSD(m, sortedTabsDict):
    """
    :param m: The input molecule to analyze.
    :param sortedTabsDict: A dictionary containing sorted TABS data.
    :returns: A dictionary containing the inter-RMSD values for each key in sortedTabsDict.
    """
    rmsds = {}
    # copying over only for debugging reasons
    molCopy = Chem.Mol(m)
    molCopy = Chem.RemoveHs(molCopy)
    for key in sortedTabsDict:
        tmp = []
        idx = sortedTabsDict[key]
        for key2 in sortedTabsDict:
            if key2 != key:
                idx2 = sortedTabsDict[key2]
                for id in idx:
                    for id2 in idx2:
                        tmp.append(rdMolAlign.GetBestRMS(molCopy, molCopy, id, id2))
        rmsds[key] = tmp
    return rmsds