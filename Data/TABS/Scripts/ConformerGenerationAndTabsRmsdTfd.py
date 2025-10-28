"""
This script generates conformers for molecules using the ETKDGv3, calculates the TABS, rmsds and tfds of the ensemble.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdmolops, TorsionFingerprints, rdMolAlign
import pickle
import uuid
import yaml
import time
import argparse
import json
from copy import deepcopy
from tabs import DihedralInfoFromTorsionLib
from tqdm import tqdm

def GetTFDBetweenAllConformers(mol, useWeights=True, maxDev='equal', symmRadius=2,
                            ignoreColinearBonds=True):
    """ Wrapper to calculate the TFD between two list of conformers 
        of a molecule

        Arguments:
        - mol:      the molecule of interest
        - useWeights: flag for using torsion weights in the TFD calculation
        - maxDev:   maximal deviation used for normalization
                    'equal': all torsions are normalized using 180.0 (default)
                    'spec':  each torsion is normalized using its specific
                            maximal deviation as given in the paper
        - symmRadius: radius used for calculating the atom invariants
                    (default: 2)
        - ignoreColinearBonds: if True (default), single bonds adjacent to
                                triple bonds are ignored
                                if False, alternative not-covalently bound
                                atoms are used to define the torsion

        Return: list of TFD values
    """
    confIds1 = [c.GetId() for c in mol.GetConformers()]
    totalNumConfs = mol.GetNumConformers()
    tl, tlr = TorsionFingerprints.CalculateTorsionLists(mol, maxDev=maxDev, symmRadius=symmRadius,
                                  ignoreColinearBonds=ignoreColinearBonds)
    torsions1 = [TorsionFingerprints.CalculateTorsionAngles(mol, tl, tlr, confId=cid) for cid in confIds1]
    tfd = []
    if useWeights:
        weights = TorsionFingerprints.CalculateTorsionWeights(mol, ignoreColinearBonds=ignoreColinearBonds)
    else:
        weights = None
    for i, t1 in enumerate(torsions1):
        for j in range(i+1, totalNumConfs):
            t2 = torsions1[j]
            tfd.append(TorsionFingerprints.CalculateTFD(t1, t2, weights=weights))
    return tfd

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
    allRmsds = rdMolAlign.GetAllConformerBestRMS(molCopy,numThreads=params['numThreads'])
    allTfds = GetTFDBetweenAllConformers(molCopy)
    f = open(f"../Output/{uuid_key}_prevMetrics.txt","w")
    f.write("cid1,cid2,tabsAgreement,rmsd,tfd\n")
    count = 0
    for i in tqdm(range(nOut)):
        for j in range(i):
            f.write(f"{i},{j},{int(allTabs[i]==allTabs[j])},{allRmsds[count]},{allTfds[count]}\n")
            count+=1
    f.close()
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformer Generation with ETKDGv3, analysis with TABS, RMSD and TFD")
    parser.add_argument("-f","--file",type=str,help="Filename of the sdf file in the Data/Input directory") 
    args = parser.parse_args()
    uuid_key = str(uuid.uuid4())
    params = yaml.load(open("../Config/Params.yml","r"),yaml.Loader)
    mol = Chem.MolFromMolFile(f"../Input/{args.file}",removeHs=False)
    ETKDGv3ConformerGeneration(mol,params,uuid_key)
    print(uuid_key)
