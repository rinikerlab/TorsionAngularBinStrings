"""
This script generates conformers for molecules using the ETKDGv3, shape aligns the conformers, and calculates the TABS and shape Tanimoto
"""

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolAlign, rdMolTransforms, rdShapeAlign
from rdkit.Geometry import Point3D
import pickle
import uuid
import yaml
import time
import argparse
import json
from copy import deepcopy
from tabs import DihedralInfoFromTorsionLib
from tqdm import tqdm
import numpy as np

def GetFailureCauses(counts):
    failureDict = {}
    for i, k in enumerate(rdDistGeom.EmbedFailureCauses.names):
        failureDict[k] = counts[i]
    return failureDict

def ETKDGv3ConformerGeneration(mol, params, uuid_key):
    pickleOutputFilename = f"../Output/{uuid_key}.pkl"
    metadataFilename = f"../Output/{uuid_key}.json"
    ps = AllChem.ETKDGv3()
    ps.useBasicKnowledge = params['useBasicKnowledge']
    ps.useSmallRingTorsions = params['useSmallRingTorsions']
    ps.useMacrocycleTorsions = params['useMacrocycleTorsions']
    ps.randomSeed = params['randomSeed']
    ps.pruneRmsThresh = params['pruneRmsThresh']
    ps.numThreads = params['numThreads']
    ps.trackFailures = True
    n = params["n"]
    start = time.time_ns()
    AllChem.EmbedMultipleConfs(mol,n,ps)
    end = time.time_ns()
    elapsed = end - start
    counts = ps.GetFailureCounts()
    allMetadata = GetFailureCauses(counts)
    allMetadata["TIME"] = elapsed
    nOut = mol.GetNumConformers()
    allMetadata["NCONFS"] = nOut
    with open(pickleOutputFilename,"wb") as pickleObject:
        pickle.dump(mol,pickleObject)
    with open(metadataFilename,"w") as f:
        json.dump(allMetadata,f)
    # calculate all TABS
    info = DihedralInfoFromTorsionLib(mol)
    allTabs = info.GetTABS()
    molCopy = deepcopy(mol)
    molCopy = Chem.RemoveHs(molCopy)
    f = open(f"../Output/{uuid_key}_metrics.txt","w")
    f.write("cid1,cid2,tabsAgreement,rmsd,shapeTanimoto,shapeColor\n")
    count = 0
    confPairs = []
    for i in tqdm(range(nOut)):
        for j in range(i):
            confPairs.append([i,j])
            m1 = Chem.Mol(mol,confId=i)
            m2 = Chem.Mol(mol,confId=j)
            m1 = Chem.RemoveAllHs(m1)
            m2 = Chem.RemoveAllHs(m2)
            opts1 = rdShapeAlign.ShapeInputOptions()
            opts2 = rdShapeAlign.ShapeInputOptions()
            cm1 = m1.GetConformer(i)
            cm2 = m2.GetConformer(j)
            tmp1 = []
            for atom, rank in zip(m1.GetAtoms(), Chem.rdmolfiles.CanonicalRankAtoms(m1, breakTies=False)):
                tmp1.append((rank, cm1.GetAtomPosition(atom.GetIdx()), 1.0))
            opts1.customFeatures = tmp1
            tmp2 = []
            for atom, rank in zip(m2.GetAtoms(), Chem.rdmolfiles.CanonicalRankAtoms(m2, breakTies=False)):
                tmp2.append((rank, cm2.GetAtomPosition(atom.GetIdx()), 1.0))
            opts2.customFeatures = tmp2
            rdMolTransforms.CanonicalizeConformer(m1.GetConformer())
            rdMolTransforms.CanonicalizeConformer(m2.GetConformer())
            tani, color = rdShapeAlign.AlignMol(m1, m2, opts1, opts2, opt_param=0.0, max_preiters=30, max_postiters=30)
            rmsdShape = rdMolAlign.CalcRMS(m1,m2,i,j)
            count+=1
            f.write(f"{i},{j},{int(allTabs[i]==allTabs[j])},{rmsdShape},{tani},{color}\n")
    f.close()
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformer Generation with ETKDGv3, analysis with TABS, RMSD and TFD")
    parser.add_argument("-f","--file",type=str,help="Filename of the sdf file in the Data/Input directory") 
    args = parser.parse_args()
    uuid_key = str(uuid.uuid4())
    params = yaml.load(open("../Config/Params.yml","r"),yaml.Loader)
    mol = Chem.MolFromMolFile(f"../Input/{args.file}",removeHs=False)
    print(uuid_key)
    ETKDGv3ConformerGeneration(mol,params,uuid_key)
