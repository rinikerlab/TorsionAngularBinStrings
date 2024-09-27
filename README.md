(T)orsion (A)ngular (B)in (S)trings
==============================================================
[![DOI](https://zenodo.org/badge/843293558.svg)](https://zenodo.org/doi/10.5281/zenodo.13384005) 
![Test Image](https://github.com/rinikerlab/TorsionAngularBinStrings/blob/main/TOC.jpg)

## Publication
(preprint) [https://doi.org/10.26434/chemrxiv-2024-b7sxm](https://doi.org/10.26434/chemrxiv-2024-b7sxm)

## Abstract

Molecular flexibility is a commonly used, but not easily quantified term. 
It is at the core of understanding composition and size of a conformational ensemble and contributes to many molecular properties.
For many computational workflows, it is necessary to reduce a conformational ensemble to meaningful representatives, however defining them and guaranteeing the ensemble's completeness is difficult.
We introduce the concepts of Torsion Angular Bin Strings (TABS) as a discrete vector representation of a conformer's dihedral angles and the number of possible TABS (nTABS) as an estimation for a molecule's ensemble size respectively.
Here we show that nTABS corresponds to an upper limit for the conformer space size for small molecules and compare the classification of conformer ensembles by TABS with classifications by RMSD. 
Overcoming known drawbacks like the molecular size dependency and threshold picking of the RMSD measure, TABS is shown to meaningful discretize the conformer space and hence allows e.g. for fast conformer space coverage checks.
The current proof-of-concept implementation is based on the ETKDGv3sr conformer generator and known torsion preferences extracted from small-molecule crystallographic data.


## Installation
Installation of the dependencies via conda using the provided environment.yml file:
```
conda env create -n tabs -f environment.yml
```

After installing the dependencies (above) and checking out this repo, run this command in this directory:
```
pip install --editable .
```



## Usage
```
import tabs
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.AddHs(Chem.MolFromSmiles("CCC(=O)CC"))
# check the assignment of torsion smarts, the class of torsion pattern, the assigned dihedral and the multiplicity of the torsion pattern
tabs.GetMultiplicityAllBonds(mol)
# check nTABS
tabs.GetnTABS(mol)
# embed molecule, get TABS
ps = AllChem.ETKDGv3()
AllChem.EmbedMolecule(mol,ps)
tabs.GetTABS(mol)
```

## How to contribute
If you want to contribute, please make sure that all currently provided unittests run and that new unittests are provided for any new functionalities.
Run the tests with
```
python -m unittest
```

## Data
The complete datasets used in the study can be reproduced by going to Data/TABS and Data/nTABS and running the provided conformer generation scripts as described in the respective READMEs.

## Analysis
The analysis notebooks to reproduce the plots shown in the study can be found in Analysis/.

## FAQs
*Are TABS dependent or independent on the atom order of otherwise identical molecules?*

As the atom numbering of a molecule is not canonicalized as part of the TABS code, it is possible to arrive at two different TABS for the same conformer of the same molecule if they differ in their atom ordering.
A way to resolve this would be to map the atom numbering of the two identical molecules to each other and then renumber of molecules. The RDKit provides code to do this and a usage example is provided in Demos/AtomRenumbering.ipynb. 
In general for the analysis of ensembles, it is recommended to work with one RDKit molecule and add all ensemble members as conformers. 

## Authors 
Jessica Braun

## Project status
in development
