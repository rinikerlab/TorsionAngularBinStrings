from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
import pickle
import uuid
import yaml
import time
import pandas as pd
import argparse

def ETKDGv3ConformerGeneration(mol, params, uuid_key):
    rdmolops.AssignStereochemistryFrom3D(mol)
    pickle_output_filename = f"../Output/{uuid_key}.pkl"
    pickle_object = open(pickle_output_filename,"wb")

    ps = AllChem.ETKDGv3()
    ps.enableSequentialRandomSeeds = True
    ps.useSmallRingTorsions = params['useSmallRingTorsions']
    ps.useMacrocycleTorsions = params['useMacrocycleTorsions']
    ps.randomSeed = params['randomSeed']
    ps.pruneRmsThresh = params['pruneRmsThresh']
    ps.numThreads = params['numThreads']
    n = params["n"]
    start = time.time_ns()
    AllChem.EmbedMultipleConfs(mol,n,ps)
    end = time.time_ns()
    elapsed = end - start
    print(f"Time elapsed: {elapsed}",flush=True)
    print("Finished embedding, pickling",flush=True)
    pickle.dump(mol,pickle_object)
    pickle_object.flush()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformer Generation with ETKDGv3")
    parser.add_argument("-f","--file",type=str,help="Filename of the sdf file in the Data/Input directory") 
    args = parser.parse_args()
    uuid_key = str(uuid.uuid4())
    params = yaml.load(open("../Config/Params.yml","r"),yaml.Loader)
    mol = Chem.MolFromMolFile(f"../Input/{args.file}",removeHs=False)
    ETKDGv3ConformerGeneration(mol,params,uuid_key)
    print(uuid_key)
