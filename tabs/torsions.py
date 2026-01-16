# Copyright (C) 2026 ETH Zurich, Jessica Braun, and other TABS contributors.
# All rights reserved.
# This file is part of TABS.
# The contents are covered by the terms of the MIT license
# which is included in the file LICENSE.

from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms
import warnings
from copy import deepcopy
import numpy as np
import enum
import json
import pathlib
from .fits import FitFunc
from .symmetry import GetTABSPermutations

# warning behaviour: always show all user warnings
warnings.simplefilter("always", UserWarning)


# the maximum value nTABS can take
MAXSIZE = 1000000
# the empirical values from ring studies (see https://doi.org/10.1021/acs.jcim.4c01513)
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

class TorsionType(enum.IntEnum):
    REGULAR = 1
    SMALL_RING = 2
    MACROCYCLE = 3
    ADDITIONAL_ROTATABLE_BOND = 4
    USER_DEFINED = 5


class TorsionLibEntry:
    def __init__(self, bounds, coeffs, fitFunc, torTyp=TorsionType.USER_DEFINED):
        self.bounds = bounds
        self.coeffs = coeffs
        self.fitFunc = fitFunc
        self.torsionType = torTyp

    def __str__(self):
        return f"{self.bounds}, {self.coeffs}, {self.fitFunc}, {self.torsionType}"
    
    def __repr__(self):
        return f"{self.bounds}, {self.coeffs}, {self.fitFunc}, {self.torsionType}"

TORSION_INFO = {}

def _LoadTorsionLibFiles():
    for ttyp,fn in ( (TorsionType.REGULAR, 'torsionPreferences_v2_regular.json'),
                     (TorsionType.SMALL_RING, 'torsionPreferences_v2_smallring.json'),
                     (TorsionType.MACROCYCLE, 'torsionPreferences_v2_macrocycle.json'),
                     (TorsionType.ADDITIONAL_ROTATABLE_BOND, 'torsionPreferences_v2_fallback.json'),
                   ):
        if ttyp not in TORSION_INFO:
            with open(str(pathlib.Path(__file__).parent.resolve().joinpath('TorsionPreferences',fn))) as f:
                INFO_FILE = json.load(f)
            TORSION_INFO[ttyp] = {}
            for key in INFO_FILE:
                TORSION_INFO[ttyp][key] = TorsionLibEntry(INFO_FILE[key]['bounds'],INFO_FILE[key]['params'],getattr(FitFunc,INFO_FILE[key]['fitFunc']), ttyp)

_LoadTorsionLibFiles()

class DihedralInfo:
    """
    stores multiple properties for one dihedral
    """

    def __init__(self, s, tt, bounds, indices=None, coeffs=None, fitFunc=None):
        self.smarts = s
        self.torsionType = tt
        self.bounds = bounds
        self.coeff = coeffs
        self.fitFunc = fitFunc
        self.indices = indices

    @property
    def multiplicity(self):
        return max(len(self.bounds), 1)
    

class DihedralsInfo:
    """
    Stores multiple properties for each dihedral.

    Instance attributes:

    :ivar molTemplate: RDKit molecule template
    :ivar undefinedStereo: bool, True if the molecule has chiral centers with undefined stereo
    :ivar smarts: list of SMARTS strings for each dihedral
    :ivar torsionTypes: list of TorsionType for each dihedral
    :ivar indices: list of atom indices for each dihedral, shape (nDihedrals, 4)
    :ivar coeffs: list of coefficients for each dihedral
    :ivar fitFuncs: list of FitFunc for each dihedral
    :ivar bounds: list of bounds for each dihedral, shape (nDihedrals, nBins)

    Instance properties:
    
    :ivar nDihedrals: int, number of dihedrals
    :ivar multiplicities: list of multiplicities for each dihedral
    :ivar nRegularTorsions: int, number of regular torsions
    :ivar nSmallRingTorsions: int, number of small ring torsions
    :ivar nMacroCycleTorsions: int, number of macrocycle torsions
    :ivar nAdditionalTorsions: int, number of additional rotatable bonds
    """

    def __init__(self, mol, raiseOnWarn=False):
        """
        :param mol: RDKit molecule template
        :param raiseOnWarn: Raise errors instead of issuing warnings
        :raises AssertionError: if the molecule does not have explicit hydrogens
        """
        assert not _needsHs(mol), "Molecule does not have explicit Hs. Consider calling AddHs"
        self.molTemplate = mol
        self.undefinedStereo = False
        self.raiseOnWarn = raiseOnWarn
        chiralInfo = Chem.FindMolChiralCenters(self.molTemplate, includeUnassigned=True)
        if chiralInfo:
            if all(item[1] == '?' for item in chiralInfo):
                self.undefinedStereo = True
                msg = "Molecule has chiral centers with undefined stereo"
                if self.raiseOnWarn:
                    raise ValueError(msg)
                warnings.warn(f"WARNING: {msg}", stacklevel=2)
        self.smarts = []
        self.torsionTypes = []
        self.indices = []
        """
        list of atom indices for every dihedral
        np.shape(self.indices) == (nDihedrals, 4)
        """
        self.coeffs = []
        self.fitFuncs = []
        self.bounds = []

    @property
    def nDihedrals(self):
        return len(self.smarts)
    @property
    def multiplicities(self):
        return [self.multiplicity(i) for i in range(self.nDihedrals)]
    @property
    def nRegularTorsions(self):
        return self.torsionTypes.count(TorsionType.REGULAR)
    @property
    def nSmallRingTorsions(self):
        return self.torsionTypes.count(TorsionType.SMALL_RING)
    @property
    def nMacroCycleTorsions(self):
        return self.torsionTypes.count(TorsionType.MACROCYCLE)
    @property
    def nAdditionalTorsions(self):
        return self.torsionTypes.count(TorsionType.ADDITIONAL_ROTATABLE_BOND)

    def append(self, tInfo):
        self.smarts.append(tInfo.smarts)
        self.torsionTypes.append(tInfo.torsionType)
        self.indices.append(tInfo.indices)
        self.coeffs.append(tInfo.coeff)
        self.bounds.append(tInfo.bounds)
        self.fitFuncs.append(tInfo.fitFunc)

    def multiplicity(self, indx):
        return max(len(self.bounds[indx]), 1)
  
    def GetConformerTorsions(self):
        """
        Calculate torsion angles for all conformers of the molecule.
        Computes the dihedral angles (torsions) for each conformer of the 
        molecule based on the provided indices. 
        The angles in range [0, 2π].

        :raises ValueError: If the molecule has no conformers.
        :return: 2D np array where rows correspond to conformers and columns 
                 correspond to dihedral angles.
        """
        if self.molTemplate.GetNumConformers() == 0:
            raise ValueError("No conformers found in molecule.")
        cids = [c.GetId() for c in self.molTemplate.GetConformers()]
        confTorsions = []
        for d in self.indices:
            tmp = []
            for cid in cids:
                conformer = self.molTemplate.GetConformer(cid)
                val = rdMolTransforms.GetDihedralRad(conformer, d[0], d[1], d[2], d[3])
                if val < 0:
                    val += 2 * np.pi
                tmp.append(val)
            confTorsions.append(tmp)
        # rows: conformers, columns: dihedrals
        return np.array(confTorsions).T
    
    def GetTABS(self, confTorsions=None, raiseOnWarn=None):
        """
        Compute the Torsion Angular Bin Strings (TABS).
        This method calculates the TABS for each conformer of the molecule based on 
        the torsion angles and predefined bounds. If no conformer torsions are provided, 
        they will be directly calculated from the conformers.

        :param confTorsions: (optional) Precalculated list of torsion angles for each conformer.
        :param raiseOnWarn: (optional) Allow overwrite of self.raiseOnWarn, default for the class
                            attribute is False.
        :raises ValueError: If no conformers are found in the molecule and no 
                            torsion angles are provided.
        :raises ValueError: If raiseOnWarn and bounds for a dihedral are not sorted.
        :return: A list of TABS for each conformer.
        """

        if confTorsions is None and self.molTemplate.GetNumConformers() == 0:
            raise ValueError("No conformers found in molecule.")
        elif self.molTemplate.GetNumConformers() > 0 and confTorsions is None:
            confTorsions = self.GetConformerTorsions()

        bounds = self.bounds
        perms = GetTABSPermutations(self.molTemplate, self.indices)

        # check if bounds are sorted
        for i in range(self.nDihedrals):
            test = deepcopy(bounds[i])
            test.sort()
            if not np.array_equal(test, bounds[i]):
                msg = f"Bounds for dihedral {i} are not sorted. This may lead to incorrect TABS calculation"
                if (self.raiseOnWarn if raiseOnWarn is None else raiseOnWarn):
                    raise ValueError(msg)
                warnings.warn(f"WARNING: {msg}", stacklevel=2)
        
        confTABS = []
        for conf in confTorsions:
            confTABS.append(_GetTABSForConformer(conf, bounds, perms))
        return confTABS

    def GetnTABS(self, maxSize=1000000):
        """
        Calculate the number of possible unique TABS (nTABS) for the molecule.
        In this implementation, the nTABS calculation uses the Burnside Lemma 
        (correction for symmetry equivalents).
        It also includes the corrections for highly correlated substructures.

        :param maxSize: Fixed maximum value of nTABS. Default is 1,000,000.
        :return: nTABS
        :raises ValueError: If the calculated number of TABS is not an integer.
        """
        # do the permutation analysis to check how many subgroups there are
        ring_mult = 1
        mult = 1
        multiplicities = self.multiplicities

        perms = GetTABSPermutations(self.molTemplate, self.indices)

        ringIndices = self.molTemplate.GetRingInfo().AtomRings()
        # check for the contributions due to small/medium rings
        contributingRingsIdentified = set()
        for i, dInfo in enumerate(self):
            if dInfo.torsionType in (TorsionType.SMALL_RING, TorsionType.MACROCYCLE):
                multiplicities[i] = 1

                for ring in ringIndices:
                    if dInfo.indices[1] in ring and dInfo.indices[2] in ring:
                        if not ring in contributingRingsIdentified:
                            ring_mult *= _RingMultFromSize(len(ring))
                            contributingRingsIdentified.add(ring)
            else: 
                mult *= multiplicities[i]

        nTABS_naive = mult * ring_mult

        if len(perms) == 1:
            return min(maxSize, nTABS_naive)
        elif nTABS_naive//len(perms) > maxSize:
            return maxSize

        nTABS = _CountOrbits(multiplicities, perms) * ring_mult

        if nTABS != int(nTABS):
            raise ValueError("nTABS is not an integer")

        return int(nTABS)
    
    def __len__(self):
        return self.nDihedrals

    def __getitem__(self, indx):
        s = self.smarts[indx]
        tt = self.torsionTypes[indx]
        ba = self.bounds[indx]
        c = self.coeffs[indx]
        di = self.indices[indx]
        ff = self.fitFuncs[indx]
        return DihedralInfo(s, tt, ba, coeffs=c, indices=di, fitFunc=ff)

def DihedralInfoFromTorsionLib(mol, torsionLibs=None, raiseOnWarn=False):
    """
    build a TorsionInfoList based on the experimental torsions library

    :param mol: RDKit molecule
    :param torsionLibs: list of dictionaries with torsion information
    :param raiseOnWarn: raise errors instead of warnings
    :return: TorsionInfoList
    :raises Warning: if no dihedrals are found
    :raises ValueError: if raiseOnWarn==True and no dihedrals are found
    """
    if torsionLibs is None:
        torsionLibs = [TORSION_INFO[TorsionType.REGULAR], TORSION_INFO[TorsionType.SMALL_RING], TORSION_INFO[TorsionType.MACROCYCLE], TORSION_INFO[TorsionType.ADDITIONAL_ROTATABLE_BOND]]
    
    clsInst = ExtractTorsionInfoWithLibs(mol, torsionLibs, raiseOnWarn=raiseOnWarn)
    if clsInst.nDihedrals == 0:
        msg = "No dihedrals found"
        if raiseOnWarn:
            raise ValueError(msg)
        warnings.warn(f"WARNING: {msg}",stacklevel=2)

    return clsInst

def _needsHs(mol):
    for atom in mol.GetAtoms():
        if atom.GetTotalNumHs(includeNeighbors=False):
            return True
    return False  

def _BondsByRDKitRotatableBondDef(m):
    # strict pattern defintion taken from rdkits function calcNumRotatableBonds
    # https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Descriptors/Lipinski.cpp#L108
    strict_pattern = "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
    matches = m.GetSubstructMatches(Chem.MolFromSmarts(strict_pattern))
    return matches

def _HydrogenFilter(m,idx):
    keep = [x for x in idx if m.GetAtomWithIdx(x).GetAtomicNum() > 1]
    return keep

def _CheckIfNotConsideredAtoms(m):
    notConsideredAtoms = Chem.MolFromSmarts('[!#1;!#6;!#7;!#8;!#9;!#15;!#16;!#17;!#35;!#53]')
    return m.HasSubstructMatch(notConsideredAtoms)

def _GetAtomIdxNotConsideredAtoms(m):
    notConsideredAtoms = Chem.MolFromSmarts('[!#1;!#6;!#7;!#8;!#9;!#15;!#16;!#17;!#35;!#53]')
    matches = m.GetSubstructMatches(notConsideredAtoms)
    atomIdx = []
    for match in matches:
        atomIdx.extend(match)
    return atomIdx

def _GetNotDescribedBonds(m):
    bonds = _BondsByRDKitRotatableBondDef(m)
    notConsideredAtoms = _GetAtomIdxNotConsideredAtoms(m)
    notConsideredBonds = []
    for bond in bonds:
        if bond[0] in notConsideredAtoms or bond[1] in notConsideredAtoms:
            notConsideredBonds.append(bond)
    return notConsideredBonds
    
def _DoubleBondStereoCheck(m, dihedralIndices, bounds):
    for i, dihedral in enumerate(dihedralIndices):
        a = int(dihedral[1])
        b = int(dihedral[2])
        A = set(x.GetIdx() for x in m.GetAtomWithIdx(a).GetBonds())
        B = [x.GetIdx() for x in m.GetAtomWithIdx(b).GetBonds()]
        trialBond = m.GetBondWithIdx(A.intersection(B).pop())
        if trialBond.GetBondType() == Chem.BondType.DOUBLE:
            if not trialBond.GetStereo() in (Chem.BondStereo.STEREONONE,Chem.BondStereo.STEREOANY):
                bounds[i] = []

def _CanonicalizeTABS(tabs, permutations):
    """
    Canonicalize the TABS string based on the provided permutations. 
    Chosen canonicalization is the lexicographically smallest string.
    """
    canon = tabs
    for p in permutations:
        tmp = ""
        for i,indx in enumerate(p):
            tmp += tabs[indx-1]
            # if we are already larger than the current canonicalization, we can short circuit
            if tmp>canon[:i+1]:
                break
        # this works without checking whether or not we short circuited because
        # in python '123' < '13'
        canon = min(tmp,canon)
    return int(canon)

def _GetTABSForConformer(torsions, sortedBounds, perms):
    """
    Generate the TABS (Torsion Angular Bin String) for a single conformer.
    This function calculates the TABS representation for a conformer by 
    determining the bin index for each torsion angle based on the provided 
    sorted bounds. The resulting TABS string is then canonicalized using 
    the specified permutations.
    :param torsions: list
        The angles of the dihedrals of a single conformer.
    :param sortedBounds: list of list
        A list where each element is a list of bin angles for a dihedral, 
        sorted in ascending order.
    :param perms: list
        A list of permutations used to canonicalize the TABS string.
    :return: str
        The canonicalized TABS string for the conformer.
    """
    nDihedrals = len(sortedBounds)
    t = ""
    for i in range(nDihedrals):
        indx = np.searchsorted(sortedBounds[i], torsions[i])
        if indx == len(sortedBounds[i]): 
            indx = 0
        t += str(indx+1)
    
    return _CanonicalizeTABS(t, perms)

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

def ETKDGv3vsRotBondCheck(m, raiseOnWarn=False):
    # dict for element
    atomNumsToSymbol = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 15:'P', 16:'S', 17:'Cl', 35:'Br', 53:'I'}
    # gives back the dihedrals and patterns that are currently not treated by ETKDG
    assert not _needsHs(m), "Molecule does not have explicit Hs. Consider calling AddHs"
    ps = rdDistGeom.ETKDGv3()
    ps.verbose = False
    ps.useSmallRingTorsions = True
    ps.useMacrocycleTorsions = True
    logs = rdDistGeom.GetExperimentalTorsions(m,ps)
    bonds = set()
    for log in logs:
        tmp = (log['atomIndices'][1],log['atomIndices'][2])
        if tmp[0] > tmp[1]:
            tmp = (tmp[1],tmp[0])
        bonds.add(tmp)
    rotBondsLipinski = set()
    rotBondsLipinskiUnsorted = _BondsByRDKitRotatableBondDef(m)
    if rotBondsLipinskiUnsorted:
        # also enforce sorting here
        for bond in rotBondsLipinskiUnsorted:
            rotBondsLipinski.add(tuple(sorted(bond)))
    if rotBondsLipinski.difference(bonds):
        if not bonds:
            msg = "No ETKDG torsion library patterns matched"
            if raiseOnWarn:
                raise ValueError(msg)
            warnings.warn(f"WARNING: {msg}",UserWarning,stacklevel=2)
        else:
            # check which bonds already considered by ETKDG
            rotBondsLipinski = rotBondsLipinski.difference(bonds)
        dihedrals = []
        patterns = []
        # define dihedrals:
        # define heavy atom neighbours
        # take the neighbours with the smallest atomIndex
        for rotBond in rotBondsLipinski:
            aid1 = [x.GetIdx() for x in m.GetAtomWithIdx(rotBond[0]).GetNeighbors()]
            aid1.remove(rotBond[1])
            aid1 = _HydrogenFilter(m,aid1)
            if aid1:
                aid1 = min(aid1)
            else:
                continue
            aid2 = [x.GetIdx() for x in m.GetAtomWithIdx(rotBond[1]).GetNeighbors()]
            aid2.remove(rotBond[0])
            aid2 = _HydrogenFilter(m,aid2)
            if aid2:
                aid2 = min(aid2)
            else:
                continue
            paid1 = m.GetAtomWithIdx(aid1).GetAtomicNum()
            paid2 = m.GetAtomWithIdx(rotBond[0]).GetAtomicNum()
            paid3 = m.GetAtomWithIdx(rotBond[1]).GetAtomicNum()
            paid4 = m.GetAtomWithIdx(aid2).GetAtomicNum()
            if paid1 in atomNumsToSymbol and paid2 in atomNumsToSymbol and paid3 in atomNumsToSymbol and paid4 in atomNumsToSymbol:
                dihedral = f"{aid1} {rotBond[0]} {rotBond[1]} {aid2}"
                dihedrals.append(dihedral)
                pattern = f"{atomNumsToSymbol[paid1]} {atomNumsToSymbol[paid2]} {atomNumsToSymbol[paid3]} {atomNumsToSymbol[paid4]}"
                patterns.append(pattern)
        if not dihedrals:
            return
        return zip(dihedrals, patterns)

def ExtractTorsionInfo(m, raiseOnWarn=False):
    return ExtractTorsionInfoWithLibs(m, [TORSION_INFO[TorsionType.REGULAR], TORSION_INFO[TorsionType.SMALL_RING], TORSION_INFO[TorsionType.MACROCYCLE]], raiseOnWarn=raiseOnWarn)

def ExtractTorsionInfoWithLibs(m, libs, raiseOnWarn=False):
    assert not _needsHs(m), "Molecule does not have explicit Hs. Consider calling AddHs"
    if _CheckIfNotConsideredAtoms(m):
        msg = ("Any torsions with atoms containing anything but H, C, N, O, F, Cl, Br, I, S or P are not considered. \n"
               "This is likely to result in an underestimation of nTABS.\n"
               f"Bonds not considered: {_GetNotDescribedBonds(m)}")
        if raiseOnWarn:
            raise ValueError(msg)
        warnings.warn(f"\nWARNING: {msg}", UserWarning, stacklevel=2)

    ps = rdDistGeom.ETKDGv3()
    ps.verbose = False
    ps.useSmallRingTorsions = True
    ps.useMacrocycleTorsions = True

    dihedrals = rdDistGeom.GetExperimentalTorsions(m,ps)
    addDihedrals = ETKDGv3vsRotBondCheck(m, raiseOnWarn=raiseOnWarn)
    if addDihedrals:
        addDihedrals, _ = zip(*addDihedrals)
    else:
        addDihedrals = []

    torsionList = DihedralsInfo(m, raiseOnWarn=raiseOnWarn)

    for log in dihedrals:
        s = log["smarts"]
        di = list(log["atomIndices"])
        
        found = False
        for lib in libs:
            if s in lib:
                entry = lib[s]
                bAngles = np.array(entry.bounds) * np.pi / 180
                tInfo = DihedralInfo(s, entry.torsionType, bAngles, indices=di, coeffs=entry.coeffs, fitFunc=entry.fitFunc)
                torsionList.append(tInfo)
                found = True
                break

        if not found:
            raise NameError(f"Error: unmatched pattern: {s}")
    
    for additional in addDihedrals:
        lib = TORSION_INFO[TorsionType.ADDITIONAL_ROTATABLE_BOND]
        s = "[*:1][*:2]!@;-[*:3][*:4]"
        entry = lib[s]
        bAngles = np.array(entry.bounds) * np.pi / 180
        di = [int(indx) for indx in additional.split(" ")]
        tInfo = DihedralInfo(s, entry.torsionType, bAngles, indices=di, coeffs=entry.coeffs, fitFunc=entry.fitFunc)
        torsionList.append(tInfo)
    """
    for additional in addDihedrals:
        # change this to make use of the fallback library
        s = "[*:1][*:2]!@;-[*:3][*:4]"
        tt = TorsionType.ADDITIONAL_ROTATABLE_BOND
        c = []
        ba = [30, 90, 150, 210, 270, 330]
        di = [int(indx) for indx in additional.split(" ")]
        tInfo = DihedralInfo(s, tt, ba, indices=di, coeffs=c)
        torsionList.append(tInfo)
    """
    # check results against what would be matched by rdkit rotatable bond definition

    _DoubleBondStereoCheck(m, torsionList.indices, torsionList.bounds)
    return torsionList