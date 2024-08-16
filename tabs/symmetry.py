from rdkit import Chem
import numpy as np

def GetTorsionPermutations(mol,dihedrals):
    ## per bond analysis
    assert len(dihedrals)>0, "no experimental torsions mapped to this molecule"
    mol = Chem.RemoveHs(mol)
    matches = mol.GetSubstructMatches(mol,useChirality=True,uniquify=False)
    allPermsSeen = set()
    for match in matches:
        remapped = []
        for tors in dihedrals:
            tors = np.array(tors.split(" "),dtype=int)
            aid1 = match[tors[1]]
            aid2 = match[tors[2]]
            remapped.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
        allPermsSeen.add(tuple(remapped))
    return tuple(allPermsSeen)

def GetSymmetryOrder(mol, dihedrals):
    ## analysis on atoms
    matches = Chem.RemoveHs(mol).GetSubstructMatches(Chem.RemoveHs(mol),useChirality=True,uniquify=False)
    permutationArray = np.array(matches,dtype=np.int16)
    _, cols = np.shape(permutationArray)
    equivalents = []
    equivalentsSetSize = []
    for c in range(cols):
        eq = set(permutationArray[:,c])
        if eq not in equivalents and len(eq)>1:
            equivalents.append(eq)
            equivalentsSetSize.append(len(eq))
    # get the list of all neighbours
    neighbours = []
    for dihedral in dihedrals:
        for neighbour in mol.GetAtomWithIdx(dihedral[1]).GetNeighbors():
            index = neighbour.GetIdx()
            if index != dihedral[2] :
                neighbours.append(index)
        for neighbour in mol.GetAtomWithIdx(dihedral[2]).GetNeighbors():
            index = neighbour.GetIdx()
            if index != dihedral[2] :
                neighbours.append(index) 
    print(neighbours)
    return np.min(equivalentsSetSize), equivalents

def _find(lst, b):
    return [i for i, x in enumerate(lst) if x==b]

def _GetDictStrippedOriginal(mol):
    mapping = {}
    for atom in mol.GetAtoms():
        atom.SetProp("atomNumberOrg",str(atom.GetIdx()))
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        mapping[atom.GetIdx()] = int(atom.GetProp("atomNumberOrg"))
    return mapping

def _TranslateMatches(matches,dictStrippedOrg):
    newMatches = []
    for match in matches:
        newMatch = []
        for m in match:
            newMatch.append(dictStrippedOrg[m])
        newMatches.append(newMatch)
    return np.array(newMatches)   

def GetTABSPermutations(mol, dihedrals):
    ## analysis on atoms
    ## bond always shall be stored as smaller atom number first !!!
    # Hs have to be removed
    dictStrippedOrg = _GetDictStrippedOriginal(mol)
    matches = np.array(Chem.RemoveHs(mol).GetSubstructMatches(Chem.RemoveHs(mol),useChirality=True,uniquify=False))
    matches = _TranslateMatches(matches,dictStrippedOrg)
    ## bonds in dihedrals
    bonds = []
    for i, dihedral in enumerate(dihedrals):
        dihedral = np.array(dihedral.split(" "),dtype=int)
        bondA1 = dihedral[1]
        bondA2 = dihedral[2]
        aid = [bondA1,bondA2]
        aid.sort()
        bonds.append(aid)
    ## build the initial TABS is first entry
    tabs = []
    for match in matches:
        tmp = []
        for i, dihedral in enumerate(dihedrals):
            dihedral = np.array(dihedral.split(" "),dtype=int)
            p1 = np.where(matches[0]==dihedral[1])[0][0]
            p2 = np.where(matches[0]==dihedral[2])[0][0]
            bondA1 = match[p1]
            bondA2 = match[p2]
            aid = [bondA1,bondA2]
            aid.sort()
            idx = _find(bonds,aid)
            tmp.append(idx[0]+1)
        tabs.append(tmp)
    # make sure that all entries are unique
    tabs = [list(x) for x in set(tuple(x) for x in tabs)]
    return tabs

def FixateRingsInPermutations(permutations, types):
    touchedUpPermutations = set()
    pArray = np.array([np.array(xi) for xi in permutations])
    for j, c in enumerate(pArray[0,:]):
        if types[c-1] == "sr":
            pArray[:,j] = c
    for i in range(pArray.shape[0]):
        tmp = tuple(pArray[i,:])
        touchedUpPermutations.add(tmp)
    touchedUpPermutations = [list(tmp) for tmp in touchedUpPermutations]
    return touchedUpPermutations