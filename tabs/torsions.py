from rdkit import Chem
from rdkit.Chem import rdDistGeom
import warnings
import copy
import numpy as np
import enum
from scipy.ndimage import gaussian_filter1d
import json
import pathlib

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

class FitFunc(enum.IntEnum):
    COS   = 1,
    GAUSS = 2,

    def call(self, params, x):
        period = 2*np.pi

        if self == FitFunc.COS:
            y = _ffitnew(x, *params)
            y /= max(y)
            return y

        elif self == FitFunc.GAUSS:
            params = np.reshape(params, (-1, 3))
            y = np.zeros(len(x))

            for p in params:
                a = p[0]
                b = p[1]
                c = p[2]

                diff = np.mod(x - b + period/2, period) - period/2
                y += a * np.exp(-((diff) / c)**2)
            return y

        else: return None

class TorsionLibEntry:
    def __init__(self, bounds, params, fitFunc, torTyp=TorsionType.USER_DEFINED):
        self.bounds = bounds
        self.params = params
        self.fitFunc = fitFunc
        self.torsionType = torTyp

    def __str__(self):
        return f"{self.bounds}, {self.params}, {self.fitFunc}, {self.torsionType}"
    def __repr__(self):
        return f"{self.bounds}, {self.params}, {self.fitFunc}, {self.torsionType}"

def _LoadTorsionLibFiles():
    global REGULAR_INFO
    global SMALLRING_INFO
    global MACROCYCLE_INFO
    global FALLBACK_INFO

    mapping = {"COS": FitFunc.COS, "GAUSS": FitFunc.GAUSS}

    if REGULAR_INFO is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_v2_regular.json'))) as f:
            REGULAR_INFO_FILE = json.load(f)
        REGULAR_INFO = {}
        for key in REGULAR_INFO_FILE.keys():
            REGULAR_INFO[key] = TorsionLibEntry(REGULAR_INFO_FILE[key]['bounds'],REGULAR_INFO_FILE[key]['params'],mapping[REGULAR_INFO_FILE[key]['fitFunc']], TorsionType.R)

    if SMALLRING_INFO is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_v2_smallring.json'))) as f:
            SMALLRING_INFO_FILE = json.load(f)
        SMALLRING_INFO = {}
        for key in SMALLRING_INFO_FILE.keys():
            SMALLRING_INFO[key] = TorsionLibEntry(SMALLRING_INFO_FILE[key]['bounds'],SMALLRING_INFO_FILE[key]['params'],mapping[REGULAR_INFO_FILE[key]['fitFunc']], TorsionType.SR)
        
    if MACROCYCLE_INFO is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_v2_macrocycle.json'))) as f:
            MACROCYCLE_INFO_FILE = json.load(f)
        MACROCYCLE_INFO = {}
        for key in MACROCYCLE_INFO_FILE.keys():
            MACROCYCLE_INFO[key] = TorsionLibEntry(MACROCYCLE_INFO_FILE[key]['bounds'],MACROCYCLE_INFO_FILE[key]['params'],mapping[REGULAR_INFO_FILE[key]['fitFunc']], TorsionType.MC)

_LoadTorsionLibFiles()


class TorsionInfo:
    """
    stores mutliple properties for one dihedral
    """

    def __init__(self, s, tt, binAngles, indices=None, coeffs=None, fitFunc=None):
        self.smarts = s
        self.torsionType = tt
        self.binAngles = binAngles
        self.coeff = coeffs
        self.fitFunc = fitFunc
        self.indices = indices

    @property
    def multiplicity(self):
        return max(len(self.binAngles), 1)
    

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
        self.binAngles = []
        #self.multiplicities = []

    def append(self, tInfo):
        self.smarts.append(tInfo.smarts)
        self.torsionTypes.append(tInfo.torsionType)
        self.indices.append(tInfo.indices)
        self.coeffs.append(tInfo.coeff)
        self.binAngles.append(tInfo.binAngles)
        self.fitFuncs.append(tInfo.fitFunc)
        #self.multiplicities.append(tInfo.multiplicity)

    @classmethod
    def WithExperimentalTorsions(cls, mol):
        """
        build a TorsionInfoList based on the experimental torsions library
        """
        return TorsionInfoList.WithTorsionLibs(mol, [SMALLRING_INFO, MACROCYCLE_INFO, REGULAR_INFO])

    @classmethod
    def WithTorsionLibs(cls, mol, torsionLibs=[]):
        """
        build a TorsionInfoList where the dihedral information is provided by a torsion library
        """
        cls = ExtractTorsionInfoWithLibs(mol, torsionLibs)

        if cls.nDihedrals == 0: warnings.warn("WARNING: no dihedrals found")
        return cls


    @classmethod
    def WithTorsions(cls, mol, dihedralIndices, confTorsions, **kwargs):
        """
        returns a TorsionInfoList with binAngles and fit coefficients based on the provided torsions
        """
        cls = TorsionInfoList(mol)
        nDihedrals = len(dihedralIndices)
        cls.indices = dihedralIndices

        yHists, xHist = ComputeTorsionHistograms(confTorsions, start=0, stop=2*np.pi, step=2*np.pi/36)
        coeffs, bins = ComputeGaussianFit(xHist, yHists, **kwargs)
        cls.binAngles = bins
        cls.coeffs = coeffs

        cls.torsionTypes = [TorsionType.USER_DEFINED] * nDihedrals
        cls.smarts = [None] * nDihedrals
        cls.fitFuncs = [FitFunc.GAUSS] * nDihedrals

        return cls

    @classmethod
    def WithMirroredTorsions(cls, mol, dihedralIndices, confTorsions, **kwargs):
        """
        same as WithTorsions but with additional enforced symmetry
        """
        cls = TorsionInfoList(mol)
        nDihedrals = len(dihedralIndices)
        cls.indices = dihedralIndices

        yHists, xHist = ComputeTorsionHistograms(confTorsions, start=0, stop=2*np.pi, step=2*np.pi/36)
        coeffs, bins = ComputeMirroredGaussianFit(xHist, yHists, **kwargs)
        cls.binAngles = bins
        cls.coeffs = coeffs

        cls.torsionTypes = [TorsionType.USER_DEFINED] * nDihedrals
        cls.smarts = [None] * nDihedrals
        cls.fitFuncs = [FitFunc.MIRRORED_GAUSS] * nDihedrals

        return cls

    @classmethod
    def WithTorsionsAndExperimentalIndices(cls, mol, confTorsions, **kwargs):
        ti = TorsionInfoList.WithExperimentalTorsions(mol, confTorsions)
        res = TorsionInfoList.WithTorsions(mol, ti.indices, confTorsions)
        res.torsionTypes = ti.torsionTypes
        pass
