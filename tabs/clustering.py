import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms
from functools import cached_property
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from mdtraj import lprmsd
from scipy.stats import chi2_contingency


class ClusterPreparation:
    """
    Class for preparing clusters of molecular conformations based on dihedral angles.

    Instance attributes:
    
    :ivar mol: RDKit molecule with conformers
    :ivar info: Dihedral information object containing indices of dihedrals
    :ivar coords: 3D coordinates of conformers
    :ivar customProfiles: Custom torsion profiles for the conformers
    :ivar m: Number of random samples to determine centroids
    :ivar centroids: Dictionary to store centroid conformations for each cluster
    :ivar centroidDistances: Distance matrix between centroids
    :ivar traj: Trajectory object for the conformers
    
    Instance properties:

    :ivar labels: TABS labels for each conformer based on dihedral angles
    :ivar uniqueLabels: Unique TABS labels present in the dataset
    :ivar uniqueLabelsCounts: Counts of each unique TABS label
    :ivar uniqueLabelsDict: Dictionary mapping each unique TABS label to the indices of conform
    """
    def __init__(self, mol, DihedralInfo, coords, customProfiles, traj=None):
        self.mol = mol
        self.info = DihedralInfo
        self.coords = coords
        self.customProfiles = customProfiles
        self.m = 100 # number of random samples to determine centroids
        self.centroids = dict()
        self.centroidDistances = np.array([])
        self.traj = traj

    @cached_property
    def labels(self):
        return self.info.GetTABS(confTorsions=self.customProfiles)
    
    @cached_property
    def uniqueLabels(self):
        uLabels = np.unique(self.labels)
        uLabels = list(uLabels)
        uLabels = [int(x) for x in uLabels]
        return uLabels

    @cached_property
    def uniqueLabelsCounts(self):
        _, uLabelsCounts = np.unique(self.labels, return_counts=True)
        return uLabelsCounts 

    @cached_property
    def uniqueLabelsDict(self):
        uLabelsDict = dict()
        for label in self.uniqueLabels:
            indices = np.argwhere(np.array(self.labels) == label).tolist()
            indices = [x[0] for x in indices]
            uLabelsDict[label] = indices
        return uLabelsDict

    def _GetCentroidWithRandomSampling(self, uLabel):
        if len(self.info.indices) < self.m:
            m = len(self.info.indices)
        else:
            m = self.m
        sampledIndices = np.random.choice(self.uniqueLabelsDict[uLabel], m, replace=True)
        rmsdMatrix = np.zeros((m, m))
        for i in range(m):
            for j in range(i+1,m):
                tmpMol1 = Chem.Mol(self.mol)
                conf1 = Chem.Conformer(tmpMol1.GetNumAtoms())
                tmpCoords1 = self.coords[sampledIndices[i], :, :].astype(np.float64) * 10
                conf1.SetPositions(tmpCoords1)
                tmpMol1.AddConformer(conf1, assignId=True)
                tmpMol1 = Chem.RemoveAllHs(tmpMol1)
                tmpMol2 = Chem.Mol(self.mol)
                conf2 = Chem.Conformer(tmpMol2.GetNumAtoms())
                tmpCoords2 = self.coords[sampledIndices[j], :, :].astype(np.float64) * 10
                conf2.SetPositions(tmpCoords2)
                tmpMol2.AddConformer(conf2, assignId=True)
                tmpMol2 = Chem.RemoveAllHs(tmpMol2)
                rmsd = rdMolAlign.GetBestRMS(tmpMol1, tmpMol2)
                rmsdMatrix[i,j] = rmsd
                rmsdMatrix[j,i] = rmsd
        sumRMSD = np.sum(rmsdMatrix, axis=1)
        minIndex = np.argmin(sumRMSD)
        centroidIndex = sampledIndices[minIndex]
        centroid = Chem.Mol(self.mol)
        conf = Chem.Conformer(self.mol.GetNumAtoms())
        tmpCoords = self.coords[centroidIndex, :, :].astype(np.float64) * 10
        conf.SetPositions(tmpCoords)
        centroid.AddConformer(conf, assignId=True)
        centroid = Chem.RemoveAllHs(centroid)
        return centroid
    
    def _GetCentroidThroughDistancesToPeaks(self, uLabel):
        clusterIndices = self.uniqueLabelsDict[uLabel]
        n = len(clusterIndices)
        m = len(self.info.indices)
        angleVals = np.zeros((n, m), dtype=np.float64)
        for i, idx in enumerate(clusterIndices):
            tmpMol = Chem.Mol(self.mol)
            conf = Chem.Conformer(tmpMol.GetNumAtoms())
            tmpCoords = self.coords[idx].astype(np.float64) * 10
            conf.SetPositions(tmpCoords)
            tmpMol.AddConformer(conf, assignId=True)
            conformer = tmpMol.GetConformer()
            for j, (a, b, c, d) in enumerate(self.info.indices):
                angleVals[i, j] = rdMolTransforms.GetDihedralRad(
                    conformer, a, b, c, d
                )
        means = angleVals.mean(axis=0)                  
        disToPeaks = np.abs(angleVals - means)           
        rowSums = disToPeaks.sum(axis=1)
        centroidIndex = clusterIndices[np.argmin(rowSums)]
        centroid = Chem.Mol(self.mol)
        conf = Chem.Conformer(self.mol.GetNumAtoms())
        conf.SetPositions(self.coords[centroidIndex].astype(np.float64) * 10)
        centroid.AddConformer(conf, assignId=True)
        return Chem.RemoveAllHs(centroid)

    def GetCentroids(self):
        """
        Calculating the centoids (representative conformations) for each unique TABS.        
        """
        jobs = [(self, label) for label in self.uniqueLabels]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(_centroid_worker_2, jobs)
        for label, centroid in results:
            self.centroids[label] = centroid
        return

    def GetDistanceMatrix(self):
        if not self.centroids:
            raise ValueError("Centroids have not been computed. Call GetCentroids() first.")
        n = len(self.centroids)
        distanceMatrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                mol1 = self.centroids[self.uniqueLabels[i]]
                mol2 = self.centroids[self.uniqueLabels[j]]
                rmsd = rdMolAlign.GetBestRMS(mol1, mol2)
                distanceMatrix[i, j] = rmsd
                distanceMatrix[j, i] = rmsd
        self.centroidDistances = distanceMatrix
        return 
    
    def VisualizeDistanceMatrix(self):
        # check if distanceMatrix is computed
        if self.centroidDistances.size == 0:
            self.centroidDistances = self.GetDistanceMatrix()
        customCmap = LinearSegmentedColormap.from_list('my_cmap', ['#0F366B', '#E4E2D4', '#D05E00'])
        plt.rcParams['font.size'] = 15
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(self.centroidDistances, cmap=customCmap, interpolation='nearest')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("customTABS Index")
        ax.set_ylabel("customTABS Index")
        ax.set_title(f"RMSD Matrix of Centroids")
        return fig

def _centroid_worker(args):
    instance, label = args
    return label, instance._GetCentroidWithRandomSampling(label)

def _centroid_worker_2(args):
    instance, label = args
    return label, instance._GetCentroidThroughDistancesToPeaks(label)

class ClusterRunner:
    def __init__(self, clusterPrep, clusterCriterion, clusterCriterionValue):
        self.clusterPrep = clusterPrep
        self.clusterCriterion = clusterCriterion
        self.clusterCriterionValue = clusterCriterionValue
        self.clustering = None
        self.clusteringDict = None
        self.nClusters = None

    @property
    def linkageMatrix(self):
        if not self.clusterPrep.centroidDistances.size:
            raise ValueError("Centroid distances not computed. Call GetDistanceMatrix() first.")
        return linkage(squareform(self.clusterPrep.centroidDistances), method='average')
    
    def GetDendrogram(self, p):
        dendrogram(self.linkageMatrix, truncate_mode='lastp', p=p)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Observations (customTABS Indices)")
        plt.ylabel("Distance")
        plt.show()

    def _GetClusterDict(self):
        if self.clustering is None:
            raise ValueError("Clustering has not been performed. Call RunClustering() first.")
        clusterDict = dict()
        for cluster in np.unique(self.clustering):
            clusterDict[int(cluster)] = np.where(self.clustering == cluster)[0]
        return clusterDict

    def RunClustering(self):
        if self.clusterCriterion == 'distance':
            clusters = fcluster(self.linkageMatrix, t=self.clusterCriterionValue, criterion='distance')
        elif self.clusterCriterion == 'maxclust':
            clusters = fcluster(self.linkageMatrix, t=self.clusterCriterionValue, criterion='maxclust')
        else:
            raise ValueError("Invalid clustering criterion. Use 'distance' or 'maxclust'.")
        self.clustering = clusters
        self.nClusters = len(np.unique(clusters))
        self.clusteringDict = self._GetClusterDict()
        return 
    
class ClusterAnalyzer:
    def __init__(self, clustering):
        self.clustering = clustering
        self.averageRmsdDisNorm = np.array([])
        self.wassersteinDis = np.array([])
        self.importances = dict()
        self.contingencyTables = dict()
        self.chi2Results = None
        self.cramersV = None

    def _GetAverageInterRmsdNorm(self):
        nClusters = self.clustering.nClusters
        clusterDict = self.clustering.clusteringDict
        averageRmsdDis = np.zeros((nClusters, nClusters))    
        distances = self.clustering.clusterPrep.centroidDistances
        for i in range(nClusters):
            for j in range(i+1, nClusters):
                clusterA = clusterDict[i+1]
                clusterB = clusterDict[j+1]
                averageRmsdDis[i,j] = np.mean([distances[x,y] for x in clusterA for y in clusterB])
                averageRmsdDis[j,i] = averageRmsdDis[i,j]
        self.averageRmsdDisNorm = averageRmsdDis / np.max(averageRmsdDis)

    def CalculateChiSquared(self):
        """
        Calculate chi-squared statistics and Cramér's V for clustering analysis.

        This method computes chi-squared test results and Cramér's V coefficient for each
        dimension in the TABS (Torsion Angular Bin Strings) dataset to assess the association
        between cluster assignments and categorical variables.

        **Method Overview**

        1. Initializes dictionaries to store chi-squared results, Cramér's V values, and contingency tables
        2. Extracts unique digits for each position across all labels
        3. Builds contingency tables for each dimension by counting occurrences of each digit within each cluster
        4. Computes chi-squared statistics and Cramér's V using :func:`scipy.stats.chi2_contingency`
        5. Handles edge cases where contingency tables have fewer than 2 rows or columns

        :returns: None

        .. rubric:: Sets Instance Attributes

        - ``chi2Results`` (dict):  
          Dictionary mapping dimension indices to tuples of  
          (chi2_statistic, p_value, degrees_of_freedom, expected_frequencies).  
          Values are ``np.nan`` for dimensions with insufficient contingency table size.

        - ``cramersV`` (dict):  
          Dictionary mapping dimension indices to Cramér's V coefficients  
          (measures association strength between 0 and 1). Values are ``np.nan`` for  
          dimensions with insufficient contingency table size.

        - ``contingencyTables`` (dict):  
          Dictionary mapping dimension indices to numpy arrays  
          representing the contingency tables (nClusters x m, where m is the number  
          of unique digits at that position).
        """
        nClusters = self.clustering.nClusters
        clusterDict = self.clustering.clusteringDict
        uLabels = self.clustering.clusterPrep.uniqueLabels
        lengthTABS = len(self.clustering.clusterPrep.info.indices)
        digitSets = dict()
        chi2Results = dict()
        cramersV = dict()
        contingencyTables = dict()

        for i in range(lengthTABS):
            chi2Results[tuple(self.clustering.clusterPrep.info.indices[i])] = None
            cramersV[tuple(self.clustering.clusterPrep.info.indices[i])] = None
            contingencyTables[tuple(self.clustering.clusterPrep.info.indices[i])] = None
            digitSets[i] = set()

        for label in uLabels:
            label = str(label)
            for i in range(lengthTABS):
                digitSets[i].add(label[i])

        for i in range(lengthTABS):
            m = len(digitSets[i])
            chiSquaredTable = np.zeros((nClusters, m))
            # get the chi-squared table
            for j in range(nClusters):
                clusterIndices = clusterDict[j+1]
                for index in clusterIndices:
                    label = str(uLabels[index])
                    digit = int(label[i])
                    chiSquaredTable[j, digit-1] += 1
            # compute chi-squared value and cramer's v
            contingencyTables[tuple(self.clustering.clusterPrep.info.indices[i])] = chiSquaredTable
            N = np.sum(chiSquaredTable)
            r, c = chiSquaredTable.shape
            if r < 2 or c < 2:
                tmp2 = (np.nan, np.nan, np.nan, np.nan)
                tmp =  np.nan
            else:
                chi2, p, dof, expected = chi2_contingency(chiSquaredTable)
                tmp2 = (chi2, p, dof, expected)
                tmp = np.sqrt(chi2 / (N * (min(r, c) - 1)))
            cramersV[tuple(self.clustering.clusterPrep.info.indices[i])] = tmp
            chi2Results[tuple(self.clustering.clusterPrep.info.indices[i])] = tmp2
        self.chi2Results = chi2Results
        self.cramersV = cramersV
        self.contingencyTables = contingencyTables