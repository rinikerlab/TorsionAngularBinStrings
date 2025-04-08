import unittest
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
try:
    from tabs import DihedralsInfo, TorsionType
except ImportError:
    raise ImportError("The tabs module is not installed. Please install it using 'pip install tabs'.")

class TestTABS(unittest.TestCase):
    mol1 = Chem.AddHs(Chem.MolFromSmiles("OCCCCC=CCCCCO"))
    mol2 = Chem.AddHs(Chem.MolFromSmiles("C1C(C)CC(CC)CC1"))
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

if __name__ == '__main__':
    unittest.main()