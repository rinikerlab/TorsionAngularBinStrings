import rdkit
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit import rdBase
import numpy as np
import math
import json
from importlib.resources import files
import warnings
import pathlib
from collections import defaultdict

# global variables for bounds and multiplicities
TPV2 = None
TPSR = None
TPMC = None
REGULAR_SMARTS_BOUNDS = None
FALLBACK_SMARTS_BOUNDS = None
SMALLRINGS_SMARTS_BOUNDS = None
MACROCYCLES_SMARTS_BOUNDS = None

def _loadAllSmartsBounds():
    # fallback should not be changed
    global FALLBACK_SMARTS_BOUNDS, REGULAR_SMARTS_BOUNDS, SMALLRINGS_SMARTS_BOUNDS, MACROCYCLES_SMARTS_BOUNDS
    global TPV2, TPSR, TPMC
    if FALLBACK_SMARTS_BOUNDS is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_fallback_smarts_bounds.txt'))) as f:
            FALLBACK_SMARTS_BOUNDS = dict(json.loads(f.read()))
    # regular torsions
    if REGULAR_SMARTS_BOUNDS is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_v2_smarts_bounds.txt'))) as f:
            REGULAR_SMARTS_BOUNDS = dict(json.loads(f.read()))
    if TPV2 is None:
        TPV2 = defaultdict(list)
        for key in REGULAR_SMARTS_BOUNDS:
            TPV2[key] = len(REGULAR_SMARTS_BOUNDS[key][1:-1].split(", "))
    # smallring torsions
    if SMALLRINGS_SMARTS_BOUNDS is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_v2_smarts_bounds_smallrings.txt'))) as f:
            SMALLRINGS_SMARTS_BOUNDS = dict(json.loads(f.read()))
    if TPSR is None:
        TPSR = defaultdict(list)
        for key in SMALLRINGS_SMARTS_BOUNDS:
            TPSR[key] = len(SMALLRINGS_SMARTS_BOUNDS[key][1:-1].split(", "))
    # macrocycle torsions
    if MACROCYCLES_SMARTS_BOUNDS is None:
        with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_v2_smarts_bounds_macrocycles.txt'))) as f:
            MACROCYCLES_SMARTS_BOUNDS = dict(json.loads(f.read()))
    if TPMC is None:
        TPMC = defaultdict(list)
        for key in MACROCYCLES_SMARTS_BOUNDS:
            TPMC[key] = len(MACROCYCLES_SMARTS_BOUNDS[key][1:-1].split(", "))

_loadAllSmartsBounds()

def _needsHs(mol):
    for atom in mol.GetAtoms():
         nHNbrs = 0
         for nbri in atom.GetNeighbors():
              if nbri.GetAtomicNum() == 1:
                  nHNbrs+=1
         if atom.GetTotalNumHs(False) > nHNbrs:
            return True
    return False  

def _movingAveragePeriodicFunction(y, window=2):
    N = len(y)
    ySmoothed = np.zeros(N)
    for i in range(N):
        tmp = 0
        tmp += y[i]
        for j in range(window):
            indexLeft = i - j
            if indexLeft < 0:
                indexLeft = N + indexLeft
            tmp += y[indexLeft]
            indexRight = i + j
            if indexRight >= N:
                indexRight = indexRight - N
            tmp += y[indexRight]
        ySmoothed[i] = tmp / (2*window + 1)
    return ySmoothed

def _derivativeFfitnew(x, s1, v1, s2, v2, s3, v3, s4, v4, s5, v5, s6, v6):
    cosx = np.cos(x)
    cosx2 = cosx*cosx
    cosx4 = cosx2*cosx2
    sinx = np.sin(x)
    return math.exp(-(v1*(1+s1*cosx) + v2*(1+s2*(2*cosx2-1)) + v3*(1+s3*(4*cosx*cosx2-3*cosx)) \
                 + v4*(1+s4*(8*cosx4-8*cosx2+1)) + v5*(1+s5*(16*cosx4*cosx-20*cosx2*cosx+5*cosx)) \
                 + v6*(1+s6*(32*cosx4*cosx2-48*cosx4+18*cosx2+1)) )) \
                 * (s1 * v1 * sinx + 4 * s2 * v2 * sinx * cosx - s3 * v3 * (3 * sinx - 12 * sinx * math.pow(cosx,2)) \
                 - s4 * v4 * (16 * sinx * cosx - 32 * sinx * math.pow(cosx,3)) - s5 * v5 * (-5 * sinx - 80 * sinx * math.pow(cosx,4) + 60 * sinx * math.pow(cosx,2)) \
                 - s6 * v6 * (-192 * sinx * math.pow(cosx,5) + 192 * sinx * math.pow(cosx,3) - 36 * sinx * cosx))

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
    
def _DoubleBondStereoCheck(m,dihedrals, multiplicities):
    for i, dihedral in enumerate(dihedrals):
        dihedral = np.array(dihedral.split(" "),dtype=int)
        a = int(dihedral[1])
        b = int(dihedral[2])
        A = set(x.GetIdx() for x in m.GetAtomWithIdx(a).GetBonds())
        B = set(x.GetIdx() for x in m.GetAtomWithIdx(b).GetBonds())
        trialBond =  m.GetBondWithIdx(list(A.intersection(B))[0])
        # Returns the type of the bond as a double (i.e. 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE)
        if trialBond.GetBondTypeAsDouble() == 2.0:
            if not trialBond.GetStereo().name in ("STEREONONE","STEREOANY"):
                multiplicities[i] = 1
    return multiplicities

def CalculateMultiplicityAndBounds(s):
    # reading in the original files with the fitted coefficients
    with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_v2_formatted.txt'))) as f: torsionPreferencesv2 = f.read()
    with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_smallrings_formatted.txt'))) as g: torsionPreferencesSmallRings = g.read()
    with open(str(pathlib.Path(__file__).parent.resolve().joinpath('torsionPreferences','torsionPreferences_macrocycles_formatted.txt'))) as h: torsionPreferencesMacrocycles = h.read()
    tpv2 = dict(json.loads(torsionPreferencesv2))
    tpsr = dict(json.loads(torsionPreferencesSmallRings))
    tpmc = dict(json.loads(torsionPreferencesMacrocycles))
    if s in tpv2: 
        all_coeffs = tpv2[s].split(" ")
    elif s in tpsr:
        all_coeffs = tpsr[s].split(" ")
    elif s in tpmc:
        all_coeffs = tpmc[s].split(" ")
    else:
        raise NameError("ERROR: unmatched pattern")
    y = [_derivativeFfitnew(j/180.0*np.pi, float(all_coeffs[0]), float(all_coeffs[1]), float(all_coeffs[2]), float(all_coeffs[3]), float(all_coeffs[4]), float(all_coeffs[5]), float(all_coeffs[6]), float(all_coeffs[7]), float(all_coeffs[8]), float(all_coeffs[9]), float(all_coeffs[10]), float(all_coeffs[11])) for j in range(0,360,1)]
    y.extend(y[0:3])
    y = _movingAveragePeriodicFunction(y,2)
    isPositive = (np.array(y)>0)*1
    bounds = np.argwhere(np.diff(isPositive) == 1)
    boundsList = []
    for b in bounds:
        candidate = b[0].tolist()
        boundsList.append(candidate)
    multiplicity = int(np.sum(np.diff(isPositive) == 1))
    return multiplicity, boundsList

def GetMultiplicityAllBonds(m):
    assert not _needsHs(m), "Molecule does not have explicit Hs. Consider calling AddHs"
    if _CheckIfNotConsideredAtoms(m):
        warnings.warn("WARNING: any torsions with atoms containing anything but H, C, N, O, F, Cl, Br, I, S or P are not considered")
    ps = rdDistGeom.ETKDGv3()
    ps.verbose = False
    ps.useSmallRingTorsions = True
    ps.useMacrocycleTorsions = True
    logs = rdDistGeom.GetExperimentalTorsions(m,ps)
    smarts = []
    dihedrals = []
    multiplicity = []
    torsiontype = []
    for log in logs:
        smarts.append(log["smarts"])
        tmp = ""
        for a in log["atomIndices"]:
            tmp += f"{a} "
        dihedrals.append(tmp[0:len(tmp)-1])
    for s in smarts:
        if s in TPV2:
            torsiontype.append("r")
            multiplicity.append(TPV2[s])
        elif s in TPSR:
            torsiontype.append("sr")
            multiplicity.append(TPSR[s])
        elif s in TPMC:
            torsiontype.append("m")
            multiplicity.append(TPMC[s])
        else:
            raise NameError("ERROR: unmatched pattern")
    # check results against what would be matched by rdkit rotatable bond definition
    additionalBonds = ETKDGv3vsRotBondCheck(m)
    if additionalBonds:
        additionalDihedrals, _ = zip(*additionalBonds)
        for additionalDihedral in additionalDihedrals:
            # the most general pattern I could come up with that fits here
            smarts.append("[*:1][*:2]!@;-[*:3][*:4]")
            torsiontype.append("arb")
            dihedrals.append(additionalDihedral)
            multiplicity.append(6)
    # another correction/check: for all double bonds, check if their stereo is defined
    # if the stereo is defined, then fix the bin
    multiplicity = _DoubleBondStereoCheck(m,dihedrals, multiplicity)
    SmartsDihedralMultiplicity = list(zip(smarts,torsiontype,dihedrals,multiplicity))
    return SmartsDihedralMultiplicity

def AnalyzeMultiplicityContributions(m):
    assert not _needsHs(m), "Molecule does not have explicit Hs. Consider calling AddHs"
    sdmList = GetMultiplicityAllBonds(m)
    _, torsiontypes, _, _ = zip(*sdmList)
    generalTorsions = 0
    smallRingTorsions = 0
    macrocycleTorsions = 0
    addtionalTorsions = 0
    for torsiontype in torsiontypes:
        if torsiontype == "r":
            generalTorsions+=1
        elif torsiontype == "sr":
            smallRingTorsions+=1
        elif torsiontype == "m":
            macrocycleTorsions+=1
        else:
            addtionalTorsions+=1
    return generalTorsions, smallRingTorsions, macrocycleTorsions, addtionalTorsions