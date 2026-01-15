Torsion Angular Bin Strings
==============================================================

[![RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
![Test Image](https://github.com/rinikerlab/TorsionAngularBinStrings/blob/main/TOC.jpg)

# Releases

**v2.0.0:** to add

**v1.0.0:** [![DOI](https://zenodo.org/badge/843293558.svg)](https://zenodo.org/doi/10.5281/zenodo.13384005) 

# Publications
[1] J. Chem. Inf. Model. 2024, DOI: [https://pubs.acs.org/doi/10.1021/acs.jcim.4c01513#](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01513#), v1.0.0

# Documentation

Our full documentation: [https://torsionangularbinstrings.readthedocs.io/en/](https://torsionangularbinstrings.readthedocs.io/en/)

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
Jessica Braun ([@brje01](https://github.com/brje01)), Djahan Lamei ([@dlamei](https://github.com/dlamei)), Greg Landrum ([@greglandrum](https://github.com/greglandrum))

# Project status
in development
