Torsion Angular Bin Strings
==============================================================
[![DOI](https://zenodo.org/badge/843293558.svg)](https://zenodo.org/doi/10.5281/zenodo.13384005) 
[![RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
![Test Image](https://github.com/rinikerlab/TorsionAngularBinStrings/blob/main/TOC.jpg)

# Publication
[1] J. Chem. Inf. Model. 2024, DOI: [https://pubs.acs.org/doi/10.1021/acs.jcim.4c01513#](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01513#)

# Abstract

Molecular flexibility is a commonly used, but not easily quantified term. 
It is at the core of understanding composition and size of a conformational ensemble and contributes to many molecular properties.
For many computational workflows, it is necessary to reduce a conformational ensemble to meaningful representatives, however defining them and guaranteeing the ensemble's completeness is difficult.
We introduce the concepts of Torsion Angular Bin Strings (TABS) as a discrete vector representation of a conformer's dihedral angles and the number of possible TABS (nTABS) as an estimation for a molecule's ensemble size respectively.
Here we show that nTABS corresponds to an upper limit for the conformer space size for small molecules and compare the classification of conformer ensembles by TABS with classifications by RMSD. 
Overcoming known drawbacks like the molecular size dependency and threshold picking of the RMSD measure, TABS is shown to meaningful discretize the conformer space and hence allows e.g. for fast conformer space coverage checks.
The current proof-of-concept implementation is based on the srETKDGv3 conformer generator and known torsion preferences extracted from small-molecule crystallographic data.


# Installation
Installation of the dependencies via conda using the provided environment.yml file:
```
conda env create -n tabs -f environment.yml
```
If you only want the minimal environment necessary (without all additional libraries for the plotting functionalities etc) use:
```
conda env create -n tabs -f minimalEnvironment.yml
```
To activate the new environment and install tabs:
```
conda activate tabs
python -m pip install git+https://github.com/rinikerlab/TorsionAngularBinStrings
```


# Usage

## TABS
With version 2 of TABS, major changes in the API were introduced.

```
from tabs import DihedralInfoFromTorsionLib
from rdkit import Chem
from rdkit.Chem import rdDistGeom

mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
# build DihedralsInfo class object
info = DihedralInfoFromTorsionLib(mol)
# check the matched SMARTS, multiplicities, torsion types
info.smarts, info.multiplicities, info.torsionTypes, info.indices
# get the number of possible TABS (calculation based on the Burnside Lemma)
info.GetnTABS()

# embed molecule, get TABS
rdDistGeom.EmbedMultipleConfs(mol, randomSeed=42, numConfs=10)
infoEnsemble = DihedralInfoFromTorsionLib(mol)
infoEnsemble.GetTABS()
```

# How to contribute
If you want to contribute, please make sure that all currently provided unittests run and that new unittests are provided for any new functionalities.
Run the tests with
```
pytest
```
If you want to deseclect the CustomTABS functionalities, run the tests with 
```
pytest -m "not custom"
```

# Data
The complete datasets used in the study can be reproduced by going to Data/TABS and Data/nTABS and running the provided conformer generation scripts as described in the respective READMEs.

# Analysis
The analysis notebooks to reproduce the plots shown in the study can be found in Analysis/.

# Authors 
Jessica Braun, Djahan Lamei, Greg Landrum

# Project status
in development
