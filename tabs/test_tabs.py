import pytest
import unittest
import numpy.testing as npt
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
try:
    from tabs import DihedralsInfo, TorsionType, DihedralInfoFromTorsionLib
    from tabs import torsions
    from tabs import custom
except ImportError:
    raise ImportError("The tabs module is not installed. Please install it.'")

class TestTABS(unittest.TestCase):
    # paths for loading test data
    current = os.getcwd()
    filePath = os.path.join(current,"Data/Tests/ensemble.pkl")
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"File not found: {filePath}")
    filePath2 = os.path.join(current,"Data/Tests/ensemble2.pkl")
    if not os.path.exists(filePath2):
        raise FileNotFoundError(f"File not found: {filePath2}")

    mol1 = Chem.AddHs(Chem.MolFromSmiles("OCCCCC=CCCCCO"))
    mol2 = Chem.AddHs(Chem.MolFromSmiles('CC[C@@H]1CCC[C@@H](C)C1'))
    mol3 = Chem.AddHs(Chem.MolFromSmiles("CS(=O)(=O)NCc1nc2cnc3[nH]ccc3c2n1[C@@H]1C[C@H]2CC[C@@H]1C2"))
    mol4 = Chem.AddHs(Chem.MolFromSmiles('CCCc1cc2cc3c(C#C[Si](CC)(CC)CC)c4cc5sc(CCC)cc5cc4c(C#C[Si](CC)(CC)CC)c3cc2s1'))
    mol5 = Chem.AddHs(Chem.MolFromSmiles('Cc1nc(N)[n+]2c(=O)cc(C)[n-]c2n1'))
    mol6 = Chem.AddHs(Chem.MolFromSmiles("CC#CC"))
    mol7 = Chem.AddHs(Chem.MolFromSmiles(r"C/C=C\C"))
    mol8 = Chem.AddHs(Chem.MolFromSmiles("CC=CC"))
    mol9 = Chem.AddHs(Chem.MolFromSmiles("C1C(C)CC(CC)CC1"))
    with open(filePath, 'rb') as f:
        mol10 = pickle.load(f)
    with open(filePath2, 'rb') as f:
        mol11 = pickle.load(f)

    def testReporter(self):
        info = DihedralInfoFromTorsionLib(self.mol1)
        self.assertEqual(info.smarts,['[$([CX3]([C])([H])):1]=[CX3:2]([H])!@;-[CH2:3][C:4]',
                                 '[$([CX3]([C])([H])):1]=[CX3:2]([H])!@;-[CH2:3][C:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[*:1][X3,X2:2]=[X3,X2:3][*:4]'])
        self.assertEqual(info.torsionTypes, [TorsionType.REGULAR,
                                TorsionType.REGULAR,
                                TorsionType.REGULAR,
                                TorsionType.REGULAR,
                                TorsionType.REGULAR,
                                TorsionType.REGULAR,
                                TorsionType.REGULAR,
                                TorsionType.REGULAR,
                                TorsionType.REGULAR])
        self.assertEqual(info.multiplicities,[3, 3, 3, 3, 3, 3, 3, 3, 2])

    def testReporterSmallRing(self):
        info = DihedralInfoFromTorsionLib(self.mol2)
        self.assertEqual(info.smarts,['[!#1:1][CX4:2]!@;-[CX4:3][!#1:4]',
                                '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]'])
        self.assertEqual(info.torsionTypes, [TorsionType.REGULAR,
                                TorsionType.SMALL_RING,
                                TorsionType.SMALL_RING,
                                TorsionType.SMALL_RING,
                                TorsionType.SMALL_RING,
                                TorsionType.SMALL_RING,
                                TorsionType.SMALL_RING])
        self.assertEqual(info.multiplicities, [3, 3, 3, 3, 3, 3, 3])

    def testReporterAdditionalTorsionContributions(self):
        info = DihedralInfoFromTorsionLib(self.mol3)
        self.assertIn(TorsionType.ADDITIONAL_ROTATABLE_BOND, info.torsionTypes)

    def testnTABS(self):
        info = DihedralInfoFromTorsionLib(self.mol1)
        nTABS = info.GetnTABS()
        self.assertEqual(nTABS, 6642)
        info = DihedralInfoFromTorsionLib(self.mol2)
        nTABS = info.GetnTABS()
        self.assertEqual(nTABS, 45)
        info = DihedralInfoFromTorsionLib(self.mol3)
        nTABS = info.GetnTABS()
        self.assertEqual(nTABS, 96)
        with self.assertWarnsRegex(UserWarning, "WARNING: No dihedrals found"):
            info = DihedralInfoFromTorsionLib(self.mol6)
        nTABS = info.GetnTABS()
        self.assertEqual(nTABS, 1)

    def testTABS(self):
        info = DihedralInfoFromTorsionLib(self.mol10)
        testingTabs = info.GetTABS()
        self.assertEqual(testingTabs, [23, 11, 22, 23, 13, 33, 33, 33, 23, 23])
        info = DihedralInfoFromTorsionLib(self.mol11)
        testingTabs = info.GetTABS()
        self.assertEqual(testingTabs, [1112111222232451121231,
                                        1112112213132451121231,
                                        1111111221222352211231,
                                        1111112122332452121231,
                                        1121111231111361211231,
                                        1111112113222352221231,
                                        1121111121121452121231,
                                        1122112121232352221231,
                                        1121112121331352221231,
                                        1122111212221352221231])

    def testNotConsideredAtomTypes(self):
        # only tests that there is a warning, not the warning message
        self.assertWarns(UserWarning,\
                            DihedralInfoFromTorsionLib,\
                            self.mol4)
        # check that the warning message is correct
        self.assertWarnsRegex(UserWarning,\
                            "WARNING: any torsions with atoms containing anything but H, C, N, O, F, Cl, Br, I, S or P are not considered.",\
                            DihedralInfoFromTorsionLib,\
                            self.mol4)

    def testNoTorsionDetected(self):
        self.assertWarnsRegex(UserWarning,\
                            "WARNING: No ETKDG torsion library patterns matched",\
                            DihedralInfoFromTorsionLib,\
                            self.mol5)
        self.assertWarnsRegex(UserWarning,\
                            "WARNING: No dihedrals found",\
                            DihedralInfoFromTorsionLib,\
                            self.mol5)
        
    def testStereoEncoding(self):
        info = DihedralInfoFromTorsionLib(self.mol7)
        self.assertEqual(info.multiplicities, [1])
        self.assertEqual(info.GetnTABS(),1)
        info = DihedralInfoFromTorsionLib(self.mol8)
        self.assertEqual(info.multiplicities, [2])
        self.assertEqual(info.GetnTABS(),2)

    def testUndefinedChiralCenter(self):
        self.assertWarnsRegex(UserWarning,\
                            "WARNING: Molecule has chiral centers with undefined stereo",\
                            DihedralInfoFromTorsionLib,\
                            self.mol9)
        
class TestCustomTABS(unittest.TestCase):
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError("MDTraj is not installed. Please install it using conda.")
    current = os.getcwd()
    filePath = os.path.join(current,"Data/Tests/traj.h5")
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"File not found: {filePath}")
    traj = md.load(filePath)
    mol = Chem.AddHs(Chem.MolFromSmiles("[H]c1c([H])c([H])c(N([H])C(=O)[C@@]2([H])C(=O)C([H])([H])C([H])([H])C([H])([H])C2([H])[H])c(C(=O)OC([H])([H])[H])c1[H]"))
    info = DihedralInfoFromTorsionLib(mol)

    @pytest.mark.custom
    def testGettingCustomProfiles(self):
        customProfiles = custom.GetTorsionProfilesFromMDTraj(self.traj, self.info.indices)        
        self.assertEqual(customProfiles.shape, (250, 11))
        binsize = np.pi*2/36
        yHists, yHistsCount, xHist = custom.ComputeTorsionHistograms(customProfiles, binsize)
        coeffs, peaks = custom.ComputeGaussianFit(xHist,yHists[4],yHistsCount[4],binsize)
        self.assertEqual(coeffs.shape, (3, 3))
        npt.assert_almost_equal(coeffs,
                                np.array([[0.42544792, 0.43633231, 0.37752448],
                                            [0.35669299, 3.14159265, 0.61328729],
                                            [0.47731244, 6.02138592, 0.42366185]]), 
                                decimal=2)
        npt.assert_almost_equal(peaks, [1.5044246510148305, 4.7787606561647555, 0.08849556770675474])
    def testGettingCustomProfiles2(self):
        customProfiles = custom.GetTorsionProfilesFromMDTraj(self.traj, self.info.indices) 
        info = DihedralInfoFromTorsionLib(self.mol, self.info.indices, customProfiles)
        npt.assert_almost_equal(info.coeffs[0],
                                np.array([[1.18175221, 6.28318531, 0.48350093]]))
if __name__ == '__main__':
    unittest.main()