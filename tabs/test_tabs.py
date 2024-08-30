import unittest
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
try:
    from . import tabs
except ImportError:
    import tabs

class TestTABS(unittest.TestCase):
    def testReporter(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCC=CCCCCO"))
        spdmList = tabs.GetMultiplicityAllBonds(mol)
        smarts, patterntypes, _, multiplicities = zip(*spdmList)
        self.assertEqual(smarts,('[$([CX3]([C])([H])):1]=[CX3:2]([H])!@;-[CH2:3][C:4]',
                                 '[$([CX3]([C])([H])):1]=[CX3:2]([H])!@;-[CH2:3][C:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[!#1:1][CX4H2:2]!@;-[CX4H2:3][!#1:4]',
                                 '[*:1][X3,X2:2]=[X3,X2:3][*:4]'))
        self.assertEqual(patterntypes,('r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'))
        self.assertEqual(multiplicities,(3, 3, 3, 3, 3, 3, 3, 3, 2))

    def testReporterSmallRing(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C1C(C)CC(CC)CC1"))
        spdmList = tabs.GetMultiplicityAllBonds(mol)
        smarts, patterntypes, _, multiplicities = zip(*spdmList)
        self.assertEqual(smarts,('[!#1:1][CX4:2]!@;-[CX4:3][!#1:4]',
                                 '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                 '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                 '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                 '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                 '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]',
                                 '[!#1;r{5-8}:1]@[CX4;r{5-8}:2]@;-[CX4;r{5-8}:3]@[!#1;r{5-8}:4]'))
        self.assertEqual(patterntypes,('r', 'sr', 'sr', 'sr', 'sr', 'sr', 'sr'))
        self.assertEqual(multiplicities,(3, 3, 3, 3, 3, 3, 3))

    def testGetnTABS(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCC=CCCCCO"))
        n = tabs.GetnTABS(mol)
        self.assertEqual(n,6642)

    def testAdditionalTorsionContributions(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CS(=O)(=O)NCc1nc2cnc3[nH]ccc3c2n1[C@@H]1C[C@H]2CC[C@@H]1C2"))
        spdmList = tabs.GetMultiplicityAllBonds(mol)
        smarts, patterntypes, _, multiplities = zip(*spdmList)
        self.assertTrue('arb' in patterntypes)
        idx = patterntypes.index('arb')
        self.assertEqual(multiplities[idx],6)
        n = tabs.GetnTABS(mol)
        self.assertEqual(n,96)

    def testNotConsideredAtomTypes(self):
        mol = Chem.AddHs(Chem.MolFromSmiles('CCCc1cc2cc3c(C#C[Si](CC)(CC)CC)c4cc5sc(CCC)cc5cc4c(C#C[Si](CC)(CC)CC)c3cc2s1'))
        self.assertWarnsRegex(UserWarning,"WARNING: any torsions with atoms containing anything but H, C, N, O, F, Cl, Br, I, S or P are not considered",tabs.GetMultiplicityAllBonds,mol)
        # self.assertWarnsRegex(UserWarning,"WARNING: torsion contained not considered atoms",tabs.GetMultiplicityAllBonds,mol)
        n = tabs.GetnTABS(mol)
        self.assertEqual(n,78)

    def testNoTorsionsDetected(self):
        mol = Chem.AddHs(Chem.MolFromSmiles('Cc1nc(N)[n+]2c(=O)cc(C)[n-]c2n1'))
        self.assertWarnsRegex(UserWarning,"WARNING: no patterns matched by ETKDG",tabs.GetMultiplicityAllBonds,mol)
        n = tabs.GetnTABS(mol)
        self.assertEqual(n,1)

    def testNoTorsionsGetTabs(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC#CC"))
        ps = AllChem.ETKDGv3()
        AllChem.EmbedMolecule(mol, ps)
        self.assertWarnsRegex(UserWarning,"WARNING: no torsions found in molecule, default of 1 returned",tabs.GetTABS,mol)
        t = tabs.GetTABS(mol)
        self.assertEqual(t,1)

    def testStereoEncoding(self):
        molStereo = Chem.AddHs(Chem.MolFromSmiles("C/C=C\C"))
        molNoStereo = Chem.AddHs(Chem.MolFromSmiles("CC=CC"))
        spdmListStereo = tabs.GetMultiplicityAllBonds(molStereo)
        _, _, _, multiplicities = zip(*spdmListStereo)
        self.assertEqual(multiplicities[0],1)
        self.assertEqual(tabs.GetnTABS(molStereo),1)
        spdmListNoStereo = tabs.GetMultiplicityAllBonds(molNoStereo)
        _, _, _, multiplicities = zip(*spdmListNoStereo)
        self.assertEqual(multiplicities[0],2)
        self.assertEqual(tabs.GetnTABS(molNoStereo),2)

if __name__ == '__main__':
    unittest.main()