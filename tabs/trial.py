import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, rdMolAlign
from collections import defaultdict


MAXSIZE = 1000000

_mediumRingsUpperBounds = {
    3: 1,
    4: 3,
    5: 11,
    6: 15,
    7: 29,
    8: 45,
    9: 115,
    10: 181,
    11: 331,
}

_macrocyclesUpperBounds = {
    12: 16549,
    13: 44934,
    14: 122002,
    15: 331251,
    16: 899394,
}

def _AnalyzeDihedralsForSymmetry(m, dihedrals):
    """
    Analyzes dihedrals for symmetry and returns exhaustive dihedrals.

    Parameters:
    m (Chem.rdchem.Mol): The molecule to analyze.
    dihedrals (list): List of dihedrals to analyze.

    Returns:
    list: List of exhaustive dihedrals.

    """
    listRanks = list(Chem.rdmolfiles.CanonicalRankAtoms(m, breakTies=False))
    exhaustiveDihedrals = []
    for tors in dihedrals:
        torsStr = tors
        tors = np.array(tors.split(" "), dtype=int)
        # get the ranks of the anchor points
        rankAid1 = listRanks[tors[0]]
        rankAid2 = listRanks[tors[3]]
        # check if there are neighbors with the same rank, if so exhaustively define all dihedrals
        neighboursAid1 = m.GetAtomWithIdx(int(tors[1])).GetNeighbors()
        neighboursAid2 = m.GetAtomWithIdx(int(tors[2])).GetNeighbors()
        symAid1 = []
        symAid2 = []
        for neighbour in neighboursAid1:
            if rankAid1 == listRanks[neighbour.GetIdx()]:
                symAid1.append(neighbour.GetIdx())
        for neighbour in neighboursAid2:
            if rankAid2 == listRanks[neighbour.GetIdx()]:
                symAid2.append(neighbour.GetIdx())
        if len(symAid1) == 1 and len(symAid2) == 2:
            exhaustiveDihedrals.append(torsStr)
        else:
            tmp = []
            for a in symAid1:
                for b in symAid2:
                    tmp.append(f"{a} {tors[1]} {tors[2]} {b}")
            exhaustiveDihedrals.append(tmp)
    return exhaustiveDihedrals

def _ApplyKnownPermutations(tabs,permutations):
    singleTabsStr = str(tabs)
    singleTabsPerms = []
    for p in permutations:
        tmp = ""
        # p = str(p)
        for i in range(len(p)):
            # why do we need the -1 here:
            # because we enumerated the bit positions in the tabs from 1
            tmp+=singleTabsStr[int(p[i])-1]
        singleTabsPerms.append(int(tmp))
    return singleTabsPerms

def AnalyzeTABSforIntraRmsd(m, sortedTabsDict):
    """
    Analyzes the Torsion Angular Bin Strings (TABS) for the intra-molecular RMSD.

    Parameters:
    - m (rdkit mol): The molecule to analyze.
    - sortedTabsDict (dict): A dictionary containing the sorted TABS.

    Returns:
    - rmsds (dict): A dictionary containing the calculated RMSDs for each key in the sortedTabsDict.

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
    Analyzes the Torsion Angular Bin Strings (TABS) for inter-RMSD values.

    Parameters:
    - m (rdkit mol): The input molecule to analyze.
    - sortedTabsDict (dict): A dictionary containing sorted TABS data.

    Returns:
    - rmsds (dict): A dictionary containing the inter-RMSD values for each key in sortedTabsDict.
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

def _CountOrbits(mults, perms):
    n_fixed_points = 0
    for perm in perms:
        n_perm_fixed_points = 1
        visited = [False] * len(perms[0])

        for p in perm:
            if not visited[p-1]:
                while not visited[p-1]:
                    visited[p-1] = True
                    p = perm[p-1]

                n_perm_fixed_points *= mults[p-1]

        n_fixed_points += n_perm_fixed_points

    return n_fixed_points / len(perms)

def _RingMultFromSize(size):
    if size < 12:
        return _mediumRingsUpperBounds[size]
    elif size < 17:
        return _macrocyclesUpperBounds[size]
    else:
        return MAXSIZE