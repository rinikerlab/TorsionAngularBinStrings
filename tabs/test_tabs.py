import unittest
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from .helpers import LoadMultipleConformerSDFile
try:
    from tabs import DihedralsInfo, TorsionType
except ImportError:
    raise ImportError("The tabs module is not installed. Please install it.'")

class TestTABS(unittest.TestCase):
    # paths for loading test data
    current = os.getcwd()
    filePath = os.path.join(current,"Data/Tests/ensemble.sdf")
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"File not found: {filePath}")

    mol1 = Chem.AddHs(Chem.MolFromSmiles("OCCCCC=CCCCCO"))
    mol2 = Chem.AddHs(Chem.MolFromSmiles('CC[C@@H]1CCC[C@@H](C)C1'))
    mol3 = Chem.AddHs(Chem.MolFromSmiles("CS(=O)(=O)NCc1nc2cnc3[nH]ccc3c2n1[C@@H]1C[C@H]2CC[C@@H]1C2"))
    mol4 = Chem.AddHs(Chem.MolFromSmiles('CCCc1cc2cc3c(C#C[Si](CC)(CC)CC)c4cc5sc(CCC)cc5cc4c(C#C[Si](CC)(CC)CC)c3cc2s1'))
    mol5 = Chem.AddHs(Chem.MolFromSmiles('Cc1nc(N)[n+]2c(=O)cc(C)[n-]c2n1'))
    mol6 = Chem.AddHs(Chem.MolFromSmiles("CC#CC"))
    mol7 = Chem.AddHs(Chem.MolFromSmiles(r"C/C=C\C"))
    mol8 = Chem.AddHs(Chem.MolFromSmiles("CC=CC"))
    mol9 = Chem.AddHs(Chem.MolFromSmiles("C1C(C)CC(CC)CC1"))
    mol10 = LoadMultipleConformerSDFile(filePath,removeHs=False)

    def testReporter(self):
        info = DihedralsInfo.FromTorsionLib(self.mol1)
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
        info = DihedralsInfo.FromTorsionLib(self.mol2)
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
        info = DihedralsInfo.FromTorsionLib(self.mol3)
        self.assertIn(TorsionType.ADDITIONAL_ROTATABLE_BOND, info.torsionTypes)

    def testnTABS(self):
        info = DihedralsInfo.FromTorsionLib(self.mol1)
        nTABS = info.GetnTABS()
        self.assertEqual(nTABS, 6642)
        info = DihedralsInfo.FromTorsionLib(self.mol2)
        nTABS = info.GetnTABS()
        self.assertEqual(nTABS, 45)
        info = DihedralsInfo.FromTorsionLib(self.mol3)
        nTABS = info.GetnTABS()
        self.assertEqual(nTABS, 96)
        with self.assertWarnsRegex(UserWarning, "WARNING: No dihedrals found"):
            info = DihedralsInfo.FromTorsionLib(self.mol6)
        nTABS = info.GetnTABS()
        self.assertEqual(nTABS, 1)

    def testTABS(self):
        print(Chem.MolToSmiles(self.mol10))
        info = DihedralsInfo.FromTorsionLib(self.mol10)
        testingTabs = info.GetTABS()
        self.assertEqual(testingTabs, [23, 11, 22, 23, 13, 33, 33, 33, 23, 23])

    def testNotConsideredAtomTypes(self):
        # only tests that there is a warning, not the warning message
        self.assertWarns(UserWarning,\
                            DihedralsInfo.FromTorsionLib,\
                            self.mol4)
        # check that the warning message is correct
        self.assertWarnsRegex(UserWarning,\
                            "WARNING: any torsions with atoms containing anything but H, C, N, O, F, Cl, Br, I, S or P are not considered.",\
                            DihedralsInfo.FromTorsionLib,\
                            self.mol4)

    def testNoTorsionDetected(self):
        self.assertWarnsRegex(UserWarning,\
                            "WARNING: No ETKDG torsion library patterns matched",\
                            DihedralsInfo.FromTorsionLib,\
                            self.mol5)
        self.assertWarnsRegex(UserWarning,\
                            "WARNING: No dihedrals found",\
                            DihedralsInfo.FromTorsionLib,\
                            self.mol5)
        
    def testStereoEncoding(self):
        info = DihedralsInfo.FromTorsionLib(self.mol7)
        self.assertEqual(info.multiplicities, [1])
        self.assertEqual(info.GetnTABS(),1)
        info = DihedralsInfo.FromTorsionLib(self.mol8)
        self.assertEqual(info.multiplicities, [2])
        self.assertEqual(info.GetnTABS(),2)

    def testUndefinedChiralCenter(self):
        self.assertWarnsRegex(UserWarning,\
                            "WARNING: Molecule has chiral centers with undefined stereo",\
                            DihedralsInfo.FromTorsionLib,\
                            self.mol9)

if __name__ == '__main__':
    unittest.main()