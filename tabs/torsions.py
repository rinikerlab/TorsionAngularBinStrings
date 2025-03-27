from rdkit import Chem
from rdkit.Chem import rdDistGeom
import warnings
import copy
import numpy as np
import enum
from scipy.ndimage import gaussian_filter1d
import json
import pathlib
import mdtraj as md
from .fits import ComputeTorsionHistograms, ComputeGaussianFit, FitFunc

#globals
REGULAR_INFO = None
SMALLRING_INFO = None
MACROCYCLE_INFO = None
FALLBACK_INFO = None

class TorsionType(enum.IntEnum):
    REGULAR = 1
    R = 1
    SMALL_RING = 2
    SR = 2
    MACRO_CYCLE = 3
    MC = 3
    ADDITIONAL_ROTATABLE_BOND = 4
    ARB = 4
    USER_DEFINED = 5

def _ffitnew(x, s1, v1, s2, v2, s3, v3, s4, v4, s5, v5, s6, v6):
    c = np.cos(x)
    c2 = c*c
    c4 = c2*c2
    return np.exp(-(v1*(1+s1*c) + v2*(1+s2*(2*c2-1)) + v3*(1+s3*(4*c*c2-3*c)) \
                    + v4*(1+s4*(8*c4-8*c2+1)) + v5*(1+s5*(16*c4*c-20*c2*c+5*c)) \
                    + v6*(1+s6*(32*c4*c2-48*c4+18*c2+1)) ))

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

def _LoadTorsionLibFiles():
    global REGULAR_INFO
    global SMALLRING_INFO
    global MACROCYCLE_INFO
    global FALLBACK_INFO

    mapping = {"COS": FitFunc.COS, "GAUSS": FitFunc.GAUSS}

    if REGULAR_INFO is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('TorsionPreferences','torsionPreferences_v2_regular.json'))) as f:
            REGULAR_INFO_FILE = json.load(f)
        REGULAR_INFO = {}
        for key in REGULAR_INFO_FILE.keys():
            REGULAR_INFO[key] = TorsionLibEntry(REGULAR_INFO_FILE[key]['bounds'],REGULAR_INFO_FILE[key]['params'],mapping[REGULAR_INFO_FILE[key]['fitFunc']], TorsionType.R)

    if SMALLRING_INFO is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('TorsionPreferences','torsionPreferences_v2_smallring.json'))) as f:
            SMALLRING_INFO_FILE = json.load(f)
        SMALLRING_INFO = {}
        for key in SMALLRING_INFO_FILE.keys():
            SMALLRING_INFO[key] = TorsionLibEntry(SMALLRING_INFO_FILE[key]['bounds'],SMALLRING_INFO_FILE[key]['params'],mapping[SMALLRING_INFO_FILE[key]['fitFunc']], TorsionType.SR)
        
    if MACROCYCLE_INFO is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('TorsionPreferences','torsionPreferences_v2_macrocycle.json'))) as f:
            MACROCYCLE_INFO_FILE = json.load(f)
        MACROCYCLE_INFO = {}
        for key in MACROCYCLE_INFO_FILE.keys():
            MACROCYCLE_INFO[key] = TorsionLibEntry(MACROCYCLE_INFO_FILE[key]['bounds'],MACROCYCLE_INFO_FILE[key]['params'],mapping[MACROCYCLE_INFO_FILE[key]['fitFunc']], TorsionType.MC)

_LoadTorsionLibFiles()


class TorsionInfo:
    """
    stores mutliple properties for one dihedral
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
    

class TorsionInfoList:
    """
    stores multiple properties for each dihedral
    """

    def __init__(self, mol):
        self.molTemplate = mol
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

    def append(self, tInfo):
        self.smarts.append(tInfo.smarts)
        self.torsionTypes.append(tInfo.torsionType)
        self.indices.append(tInfo.indices)
        self.coeffs.append(tInfo.coeff)
        self.bounds.append(tInfo.bounds)
        self.fitFuncs.append(tInfo.fitFunc)

    def multiplicity(self, indx):
        return max(len(self.bounds[indx]), 1)

    @classmethod
    def WithTorsionLibs(cls, mol, torsionLibs=None):
        """
        build a TorsionInfoList based on the experimental torsions library
        : param mol: rdkit molecule
        : param torsionLibs: list of dictionaries with torsion information
        : return: TorsionInfoList
        : raises Warning: if no dihedrals are found
        """
        if torsionLibs is None:
            torsionLibs = [REGULAR_INFO, SMALLRING_INFO, MACROCYCLE_INFO]
        
        cls = ExtractTorsionInfoWithLibs(mol, torsionLibs)
        if cls.nDihedrals == 0: warnings.warn("WARNING: no dihedrals found")

        return cls

    @classmethod
    def WithCustomTorsions(cls, mol, dihedralIndices, customTorsionProfiles, **kwargs):
        """
        returns a TorsionInfoList with bounds and fit coefficients based on the provided torsion profiles
        : param mol: rdkit molecule
        : param dihedralIndices: list of atom indices for every dihedral
        : param customTorsionProfiles: list of custom torsion profiles
        : param kwargs: additional arguments for ComputeGaussianFit
        """
        cls = TorsionInfoList(mol)
        nDihedrals = len(dihedralIndices)
        cls.indices = dihedralIndices

        yHists, xHist = ComputeTorsionHistograms(customTorsionProfiles, start=0, stop=2*np.pi, step=2*np.pi/36)
        coeffs = []
        bins = []
        for yHist in yHists:
            c, b = ComputeGaussianFit(xHist, yHist, **kwargs)
            coeffs.append(c)
            bins.append(b)
        cls.bounds = bins
        cls.coeffs = coeffs

        cls.torsionTypes = [TorsionType.USER_DEFINED] * nDihedrals
        cls.smarts = [None] * nDihedrals
        cls.fitFuncs = [FitFunc.GAUSS] * nDihedrals

        return cls

    # @classmethod
    # def WithMirroredTorsions(cls, mol, dihedralIndices, customTorsionProfiles, **kwargs):
    #     """
    #     same as WithTorsions but with additional enforced symmetry
    #     """
    #     cls = TorsionInfoList(mol)
    #     nDihedrals = len(dihedralIndices)
    #     cls.indices = dihedralIndices

    #     yHists, xHist = ComputeTorsionHistograms(customTorsionProfiles, start=0, stop=2*np.pi, step=2*np.pi/36)
    #     coeffs, bins = ComputeMirroredGaussianFit(xHist, yHists, **kwargs)
    #     cls.bounds = bins
    #     cls.coeffs = coeffs

    #     cls.torsionTypes = [TorsionType.USER_DEFINED] * nDihedrals
    #     cls.smarts = [None] * nDihedrals
    #     cls.fitFuncs = [FitFunc.MIRRORED_GAUSS] * nDihedrals

    #     return cls

    # @classmethod
    # def WithTorsionsAndExperimentalIndices(cls, mol, customTorsionProfiles, **kwargs):
    #     ti = TorsionInfoList.WithExperimentalTorsions(mol, customTorsionProfiles)
    #     res = TorsionInfoList.WithTorsions(mol, ti.indices, customTorsionProfiles)
    #     res.torsionTypes = ti.torsionTypes
    #     pass

def _needsHs(mol):
    for atom in mol.GetAtoms():
         nHNbrs = 0
         for nbri in atom.GetNeighbors():
              if nbri.GetAtomicNum() == 1:
                  nHNbrs+=1
         if atom.GetTotalNumHs(False) > nHNbrs:
            return True
    return False  

def _BondsByRdkitRotatableBondDef(m):
    # strict pattern defintion taken from rdkits function calcNumRotatableBonds
    # https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Descriptors/Lipinski.cpp#L108
    strict_pattern = "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
    matches = m.GetSubstructMatches(Chem.MolFromSmarts(strict_pattern))
    return matches

def _HydrogenFilter(m,idx):
    keep = []
    for x in idx:
        if m.GetAtomWithIdx(x).GetAtomicNum() > 1:
            keep.append(x)
    return keep

def ETKDGv3vsRotBondCheck(m):
    # dict for element
    atomNumsToSymbol = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 15:'P', 16:'S', 17:'Cl', 35:'Br', 53:'I'}
    keys = atomNumsToSymbol.keys()
    # gives back the dihedrals and patterns that are currently not treated by ETKDG
    # assert Chem.rdmolops.HasQueryHs(m)[0], "Molecule does not have explicit Hs. Consider calling AddHs"
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
        bonds.add(tuple(tmp))
    rotBondsLipinski = set()
    rotBondsLipinskiUnsorted = _BondsByRdkitRotatableBondDef(m)
    if rotBondsLipinskiUnsorted:
        # also enforce sorting here
        for bond in rotBondsLipinskiUnsorted:
            rotBondsLipinski.add(tuple(sorted(bond)))
    if rotBondsLipinski.difference(bonds):
        if not bonds:
            warnings.warn("WARNING: no patterns matched by ETKDG",UserWarning)
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
                aid1 = np.min(aid1)
            else:
                continue
            aid2 = [x.GetIdx() for x in m.GetAtomWithIdx(rotBond[1]).GetNeighbors()]
            aid2.remove(rotBond[0])
            aid2 = _HydrogenFilter(m,aid2)
            if aid2:
                aid2 = np.min(aid2)
            else:
                continue
            dihedral = f"{aid1} {rotBond[0]} {rotBond[1]} {aid2}"
            paid1 = m.GetAtomWithIdx(int(aid1)).GetAtomicNum()
            paid2 = m.GetAtomWithIdx(int(rotBond[0])).GetAtomicNum()
            paid3 = m.GetAtomWithIdx(int(rotBond[1])).GetAtomicNum()
            paid4 = m.GetAtomWithIdx(int(aid2)).GetAtomicNum()
            if paid1 in keys and paid2 in keys and paid3 in keys and paid4 in keys:
                pattern = f"{atomNumsToSymbol[paid1]} {atomNumsToSymbol[paid2]} {atomNumsToSymbol[paid3]} {atomNumsToSymbol[paid4]}"
                dihedrals.append(dihedral)
                patterns.append(pattern)
        if not dihedrals:
            return
        return zip(dihedrals, patterns)

def _CheckIfNotConsideredAtoms(m):
    notConsideredAtoms = Chem.MolFromSmarts('[!#1;!#6;!#7;!#8;!#9;!#15;!#16;!#17;!#35;!#53]')
    return m.HasSubstructMatch(notConsideredAtoms)
    
def _DoubleBondStereoCheck(m, dihedralIndices, bounds):
    #for i, dihedral in enumerate(dihedrals):
    for i, dihedral in enumerate(dihedralIndices):
        #dihedral = sdm.dihedral
        a = int(dihedral[1])
        b = int(dihedral[2])
        A = set(x.GetIdx() for x in m.GetAtomWithIdx(a).GetBonds())
        B = set(x.GetIdx() for x in m.GetAtomWithIdx(b).GetBonds())
        trialBond =  m.GetBondWithIdx(list(A.intersection(B))[0])
        # Returns the type of the bond as a double (i.e. 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE)
        if trialBond.GetBondTypeAsDouble() == 2.0:
            if not trialBond.GetStereo().name in ("STEREONONE","STEREOANY"):
                bounds[i] = []
                #multiplicities[i] = 1

def ExtractTorsionInfo(m):
    return ExtractTorsionInfoWithLibs(m, [REGULAR_INFO, SMALLRING_INFO, MACROCYCLE_INFO])

def ExtractTorsionInfoWithLibs(m, libs):
    assert not _needsHs(m), "Molecule does not have explicit Hs. Consider calling AddHs"
    if _CheckIfNotConsideredAtoms(m):
        warnings.warn("WARNING: any torsions with atoms containing anything but H, C, N, O, F, Cl, Br, I, S or P are not considered")

    ps = rdDistGeom.ETKDGv3()
    ps.verbose = False
    ps.useSmallRingTorsions = True
    ps.useMacrocycleTorsions = True

    dihedrals = rdDistGeom.GetExperimentalTorsions(m,ps)
    addDihedrals = ETKDGv3vsRotBondCheck(m)
    if addDihedrals:
        addDihedrals, _ = zip(*addDihedrals)
    else:
        addDihedrals = []

    torsionList = TorsionInfoList(m)

    for log in dihedrals:
        s = log["smarts"]
        di = list(log["atomIndices"])

        found = False
        for lib in libs:
            if s in lib:
                entry = lib[s]
                bAngles = np.array(entry.bounds) * np.pi / 180
                tInfo = TorsionInfo(s, entry.torsionType, bAngles, indices=di, coeffs=entry.coeffs, fitFunc=entry.fitFunc)
                torsionList.append(tInfo)
                found = True
                break

        if not found:
            raise NameError(f"Error: unmatched pattern: {s}")

    for additional in addDihedrals:
        s = "[*:1][*:2]!@;-[*:3][*:4]"
        tt = TorsionType.ARB
        c = []
        ba = [30, 90, 150, 210, 270, 330]
        di = [int(indx) for indx in additional.split(" ")]
        tInfo = TorsionInfo(s, tt, ba, indices=di, coeffs=c)
        torsionList.append(tInfo)

    # check results against what would be matched by rdkit rotatable bond definition

    _DoubleBondStereoCheck(m, torsionList.indices, torsionList.bounds)
    return torsionList

def GetTorsionProfilesFromMDTraj(mdtraj, torsionIndices):
    """ compute the dihedral angles
    Parameters:
    - mdtraj: trajectory object
    - torsionIndices: list of dihedral indices

    Returns:
    - np.array(shape=(nConformers, nDihedrals), dtype=float): dihedral angles in [0, 2*pi]
    """
    dAngles = md.compute_dihedrals(mdtraj, torsionIndices)
    dAngles[dAngles < 0] += 2*np.pi
    return dAngles