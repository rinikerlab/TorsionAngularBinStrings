from .multiplicity import GetMultiplicityAllBonds, _needsHs 
from .multiplicity import REGULAR_SMARTS_BOUNDS, FALLBACK_SMARTS_BOUNDS, MACROCYCLES_SMARTS_BOUNDS, SMALLRINGS_SMARTS_BOUNDS
from .symmetry import GetTABSPermutations
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, rdMolAlign
import json
import pathlib
from collections import defaultdict
import warnings


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

def _AngleConversionDegToRad(a):
    ## conversion from [0,360] to [0,2pi]
    return a*np.pi/180

def _AngleConversionRadShift(a):
    ## shift to work on [0,2pi]
    if a < 0:
        return 2*np.pi+a
    else:
        return a

def _AssignTorsionsToBins(smarts,torsionVals,torsionTypes,multiplicities):
    # get the bounds from the smarts
    tpv2b = REGULAR_SMARTS_BOUNDS
    tpfbBounds = FALLBACK_SMARTS_BOUNDS
    tpmBounds = MACROCYCLES_SMARTS_BOUNDS
    tpsrBounds = SMALLRINGS_SMARTS_BOUNDS
    singleTabs = ""
    for s, t, tt, m in zip(smarts,torsionVals,torsionTypes, multiplicities):
        if tt == "r":
            ba = tpv2b[s]
        elif tt == "arb":
            ba = tpfbBounds[s]
        elif tt == "m":
            ba = tpmBounds[s]
        elif tt == "sr":
            ba = tpsrBounds[s]
        baLen = len(ba)
        ba = ba[1:baLen-1].split(", ")
        ba = np.array(ba,dtype=int)
        ba = np.apply_along_axis(_AngleConversionDegToRad,0,ba)
        if np.shape(ba)[0] > m:
            # only temporarly set multiplicity back to not-stereo corrected value
            m = np.shape(ba)[0]
        for i in range(m):
            if t < ba[i]:
                singleTabs += f"{i+1}"
                break
            if i == m-1 and t > ba[i]:
                singleTabs += f'1'
                break
    return int(singleTabs)

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

def GetTABS(m, confId=-1):
    """
    Calculates the TABS (Torsion Angular Bin Strings) value for a given molecule and conformer.

    Parameters:
    - m (rdkit.Chem.rdchem.Mol): The molecule for which to calculate the TABS value.
    - confId (int): The ID of the conformer for which to calculate the TABS value.

    Returns:
    - int: The minimum TABS value among all possible permutations.

    """
    sdmList = GetMultiplicityAllBonds(m)
    if not sdmList:
        warnings.warn("WARNING: no torsions found in molecule, default of 1 returned")
        return 1
    else:
        smarts, patterntype, dihedrals, multiplicities = zip(*sdmList)
    conf = m.GetConformer(confId)
    torsionVals = []
    for tors in dihedrals:
        tors = np.array(tors.split(" "), dtype=int)
        # conversion here need bc numpy int not python int
        tmp = rdMolTransforms.GetDihedralRad(conf, int(tors[0]), int(tors[1]), int(tors[2]), int(tors[3]))
        torsionVals.append(_AngleConversionRadShift(tmp))
    singleTabs = _AssignTorsionsToBins(smarts, torsionVals, patterntype, multiplicities)
    # global symmetry correction by permutations
    perms = GetTABSPermutations(m, dihedrals)
    singleTabsPerms = _ApplyKnownPermutations(singleTabs, perms)
    return int(np.min(singleTabsPerms))

def GetTABSMultipleConfs(m):
    sdmList = GetMultiplicityAllBonds(m)
    smarts, patterntype, dihedrals, multiplicities = zip(*sdmList)
    confIds = []
    allConfsTabs = []
    perms = GetTABSPermutations(m,dihedrals)
    for c in m.GetConformers():
        confIds.append(c.GetId())
        torsionVals = []
        for tors in dihedrals:
            tors = np.array(tors.split(" "),dtype=int)
            tmp = rdMolTransforms.GetDihedralRad(c,int(tors[0]),int(tors[1]),int(tors[2]),int(tors[3]))
            torsionVals.append(_AngleConversionRadShift(tmp))
        singleTabs = _AssignTorsionsToBins(smarts,torsionVals,patterntype,multiplicities)
        # global symmetry correction by permutations
        singleTabsPerms =_ApplyKnownPermutations(singleTabs,perms)
        allConfsTabs.append(int(np.min(singleTabsPerms)))
    return confIds, allConfsTabs

def _addTorsionBin(tabsList,multiplicity):
    newTabsList = []
    if not tabsList:
        newTabsList = [f"{i+1}" for i in range(multiplicity)]
    else:
        for a in tabsList:
            newTabsList.extend([a+f"{i+1}" for i in range(multiplicity)])
    return newTabsList

def GetnTABS(m):
    assert not _needsHs(m), "Molecule does not have explicit Hs. Consider calling AddHs"
    sdmList = GetMultiplicityAllBonds(m)
    if not sdmList:
        return 1
    _, torsiontypes, dihedrals, multiplicities_org = zip(*sdmList)
    # do the permutation analysis to check how many subgroups there are
    permutations = GetTABSPermutations(m,dihedrals)
    # check for the contributions due to small/medium rings
    ringMultiplicities = []
    if "sr" in torsiontypes or "m" in torsiontypes:
        multiplicities = []
        ringIndices = m.GetRingInfo().AtomRings()
        contributingRingsIdentified = set()
        for type, dihedral, multiplicity in zip(torsiontypes, dihedrals, multiplicities_org):
            dihedral = tuple([int(num) for num in dihedral.split(" ")])
            if type in ("sr", "m"):
                multiplicities.append(1)
                for ring in ringIndices:
                    if dihedral[1] in ring and dihedral[2] in ring:
                        contributingRingsIdentified.add(ring)
            else:
                multiplicities.append(multiplicity)
        for ring in contributingRingsIdentified:
            if len(ring) < 12:
                ringMultiplicities.append(_mediumRingsUpperBounds[len(ring)])
            elif len(ring) < 17:
                ringMultiplicities.append(_macrocyclesUpperBounds[len(ring)])
            else:
                ringMultiplicities.append(MAXSIZE)
    else:
        multiplicities = multiplicities_org
    # get naive nTABS
    nTABS_naive = np.prod(multiplicities)
    if ringMultiplicities:
        nTABS_naive *= np.prod(ringMultiplicities)
    # check crude approximation for conformer space
    # if there is no symmetry than the naive counting is the right answer
    if len(permutations) == 1:
        if abs(nTABS_naive) > MAXSIZE:
            return MAXSIZE
        else:
            return int(nTABS_naive)
    else:
        if int(abs(nTABS_naive)/len(permutations)) > MAXSIZE:
            return MAXSIZE
    # need to write them out explicitly and then keep track of the already discovered ones
    tabsList = []
    for m in multiplicities:
        tabsList = _addTorsionBin(tabsList,m)
    # convert to set
    correctedTabsList = []
    tabsList = set(int(tab) for tab in tabsList)
    while(len(tabsList)>0):
        candidate = tabsList.pop()
        candidatePerms = _ApplyKnownPermutations(candidate,permutations)
        for drops in candidatePerms:
            tabsList.discard(drops)
        correctedTabsList.append(min(candidatePerms))
    # if it got to here, then need to multiply the ring confs back in
    nTABS = len(correctedTabsList)
    if ringMultiplicities:
        nTABS = len(correctedTabsList) * np.prod(ringMultiplicities)
    return int(nTABS)

def SortEnsembleByTABS(m):
    cIds, allTabs = GetTABSMultipleConfs(m)
    sortedByTabs = defaultdict(list)
    for i, t in enumerate(allTabs):
        sortedByTabs[int(t)].append(cIds[i])
    return sortedByTabs

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