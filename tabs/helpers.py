from rdkit import Chem

def LoadMultipleConformerSDFile(path,removeHs):
    suppl = Chem.ForwardSDMolSupplier(path,removeHs=removeHs)
    mol = next(suppl)
    for m in suppl:
        mol.AddConformer(m.GetConformer(),assignId=True)
    return mol