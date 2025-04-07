import unittest
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
try:
    from tabs import DihedralsInfo
    from tabs.torsions import TorsionType
except ImportError:
    raise ImportError("The tabs module is not installed. Please install it using 'pip install tabs'.")

class TestTABS(unittest.TestCase):
    mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCC=CCCCCO"))
    def testReporter(self):
        info = DihedralsInfo.FromTorsionLib(self.mol)
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