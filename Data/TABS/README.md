## Purpose
Generate 100 conformers for each molecule of the Platinum dataset. For all ensembles calculate the TABS, RMSD (all to all), TFD (all to all) values. 
Compare a categorization in same or not the same according to the different metrics to each other. 

## Usage
For the comparison studies in [1]:
```
cd Scripts/
python ConformerGenerationAndTabsRmsdTfd.py -f 1A0_4HXW.sdf
```
For the additional analysis comparing against the shape tanimoto of shape aligned pairs of conformers:
```
cd Scripts/
python ConformerGenerationAndShapeTabsRmsd.py -f 1A0_4HXW.sdf
```