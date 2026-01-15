# Copyright (C) 2026 ETH Zurich, Jessica Braun, and other TABS contributors.
# All rights reserved.
# This file is part of TABS.
# The contents are covered by the terms of the MIT license
# which is included in the file LICENSE.

import numpy as np
from .fits import ComputeTorsionHistograms, ComputeGaussianFit, FitFunc
from .torsions import DihedralsInfo, TorsionType
from .plots import _GridPlot
import mdtraj as md

def CustomDihedralInfo(mol, dihedralIndices, customTorsionProfiles, showFits=False, **kwargs):
    """
    Returns a TorsionInfoList with bounds and fit coefficients based on the provided torsion profiles

    :param mol: rdkit molecule
    :param dihedralIndices: list of atom indices for every dihedral
    :param customTorsionProfiles: list of custom torsion profiles
    :param kwargs: additional arguments for ComputeGaussianFit
    :param showFits: if True, plots the fits for the dihedrals
    :returns: DihedralsInfo object with the computed bounds and coefficients
    """
    clsInst = DihedralsInfo(mol)
    nDihedrals = len(dihedralIndices)
    clsInst.indices = dihedralIndices

    binsize = 2*np.pi/36
    yHists, yHistsCount, xHist = ComputeTorsionHistograms(customTorsionProfiles, binsize)
    coeffs = []
    bounds = []
    for yHist, yHistCount in zip(yHists,yHistsCount):
        c, b = ComputeGaussianFit(xHist, yHist, yHistCount, binsize, **kwargs)
        coeffs.append(c)
        bounds.append(b)
        
    # make sure that bounds are sorted
    boundsSorted = []
    for i in range(nDihedrals):
        order = np.argsort(bounds[i])
        boundsSorted.append([bounds[i][j] for j in order])

    clsInst.bounds = boundsSorted
    clsInst.coeffs = coeffs
    clsInst.torsionTypes = [TorsionType.USER_DEFINED] * nDihedrals
    clsInst.smarts = [None] * nDihedrals
    clsInst.fitFuncs = [FitFunc.GAUSS] * nDihedrals

    def _PlotProb(ax, indx):
        xFit = np.linspace(0, 2*np.pi, 2*len(xHist))
        yFit = FitFunc.GAUSS.call(clsInst.coeffs[indx], xFit)
        ax.bar(xHist, yHists[indx], width=binsize, color="lightblue", alpha=0.7)
        ax.plot(xFit, yFit, color="black")

        ba = clsInst.bounds[indx]
        for a in ba:
            ax.axvline(a, color="black")

        ax.set_xlabel("Dihedral angle / rad")
        ax.set_ylabel("Density")
        ax.set_title(f"Dihedral {indx} - {clsInst.indices[indx]}")

    if showFits:
        _GridPlot(nDihedrals, _PlotProb)

    return clsInst

def GetTorsionProfilesFromMDTraj(mdtraj, torsionIndices):
    """
    Compute the dihedral angles from an MD trajectory.

    :param mdtraj: MDTraj trajectory object containing the molecular dynamics data.
    :param torsionIndices: List of dihedral indices, where each index is a list of 
        four atom indices defining a torsion angle.
    :returns: A NumPy array of shape (nConformers, nDihedrals) containing the 
        dihedral angles in radians, adjusted to the range [0, 2*pi].
    """
    dAngles = md.compute_dihedrals(mdtraj, torsionIndices)
    dAngles[dAngles < 0] += 2*np.pi
    return dAngles