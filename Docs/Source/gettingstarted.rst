Introduction
=============

.. _GetStarted:


What is TABS?
-------------
Molecular flexibility is a commonly used, but not easily quantified term. 
It is at the core of understanding composition and size of a conformational ensemble and contributes to many molecular properties. 
For many computational workflows, it is necessary to reduce a conformational ensemble to meaningful representatives, however defining them and guaranteeing the ensemble's completeness is difficult. 
We introduce the concepts of Torsion Angular Bin Strings (TABS) as a discrete vector representation of a conformer's dihedral angles and the number of possible TABS (nTABS) as an estimation for a molecule's ensemble size respectively. 
Here we show that nTABS corresponds to an upper limit for the conformer space size for small molecules and compare the classification of conformer ensembles by TABS with classifications by RMSD. 
Overcoming known drawbacks like the molecular size dependency and threshold picking of the RMSD measure, TABS is shown to meaningful discretize the conformer space and hence allows e.g. for fast conformer space coverage checks. 
The current proof-of-concept implementation is based on the srETKDGv3 conformer generator and known torsion preferences extracted from small-molecule crystallographic data.

Installation Guide
-------------------
Installation of the dependencies via conda using the provided environment.yml file:

.. code-block:: bash

    conda env create -n tabs -f --file=https://raw.githubusercontent.com/rinikerlab/TorsionAngularBinStrings/main/environment.yml


To activate the new environment and install TABS via pip:

.. code-block:: bash

    conda activate tabs
    python -m pip install git+https://github.com/rinikerlab/TorsionAngularBinStrings

Basic Usage
------------
With version 2 of TABS, major changes in the API were introduced.

.. code-block:: python

    from tabs import DihedralInfoFromTorsionLib
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
    # build DihedralsInfo class object
    info = DihedralInfoFromTorsionLib(mol)
    # check the matched SMARTS, multiplicities, torsion types
    info.smarts, info.multiplicities, info.torsionTypes, info.indices
    # get the number of possible TABS (calculation based on Burnside's Lemma)
    info.GetnTABS()

    # embed molecule, get TABS
    rdDistGeom.EmbedMultipleConfs(mol, randomSeed=42, numConfs=10)
    infoEnsemble = DihedralInfoFromTorsionLib(mol)
    infoEnsemble.GetTABS()

