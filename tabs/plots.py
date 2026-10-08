# Copyright (C) 2026 ETH Zurich, Jessica Braun, Djahan Lamei, Enrico Ruijsenaars, Greg Landrum, and other TABS contributors.

import numpy as np
import math
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw import IPythonConsole
import sys
from tabs import fits
try: 
    from ipywidgets import interact, IntSlider
except ImportError:
    print("ipywidgets not installed. Please install it using conda.")
    sys.exit(1)
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Please install it using conda.")
    sys.exit(1)
try:
    import py3Dmol
except ImportError:
    print("py3Dmol not installed. Please install it using conda.")
    sys.exit(1)

IPythonConsole.ipython_3d = True
plt.rcParams.update({'font.size': 12})

def _GridPlot(nPlots, plotFn, plotsPerRow=4, title=None, args=None, start=0, projection=None):
    """ 
    helper function for showing multiple plots in a grid
    
    Parameters:
    - nPlots: number of plots
    - plotFn: takes an axis and index of the current plot as argument
    - plotsPerRow: int
    - title: Optional[String]
    """
    if nPlots == 0: return

    rows = (nPlots) // plotsPerRow 
    if nPlots % plotsPerRow:
        rows += 1
    fig, axes = plt.subplots(rows, plotsPerRow, figsize=(plotsPerRow * 5, rows * 5), subplot_kw=dict(projection=projection))

    axes = np.array(axes).flatten()
    for i in range(start, rows * plotsPerRow):
        ax = axes[i]
        if i < nPlots:
            if args is None:
                plotFn(ax, i)
            else:
                plotFn(ax, i, *args)
        else:
            ax.axis('off')

    if title:
        fig.suptitle(title)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3)  
    else:
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.3)
    return fig

def _ffitnew(x, s1, v1, s2, v2, s3, v3, s4, v4, s5, v5, s6, v6):
    c = np.cos(x)
    c2 = c*c
    c4 = c2*c2

    return math.exp(-(v1*(1+s1*c) + v2*(1+s2*(2*c2-1)) + v3*(1+s3*(4*c*c2-3*c)) \
                    + v4*(1+s4*(8*c4-8*c2+1)) + v5*(1+s5*(16*c4*c-20*c2*c+5*c)) \
                    + v6*(1+s6*(32*c4*c2-48*c4+18*c2+1)) ))

def PlotCosineFit(coeffs, smarts=None):
    """
    PlotCosineFit(coeffs, smarts=None)
    Plot a fitted cosine function over the range [0, 2π].

    Parameters
    ----------
    coeffs : array-like
        Coefficients passed to the cosine fit evaluator. These are forwarded to
        fits.FitFunc.COS.call(coeffs, x). 
    smarts : str, optional
        Optional label (for example a SMARTS pattern) to set as the plot title.
        If None, no title is added.

    Returns
    -------
    None
        The function creates a new matplotlib Figure and Axes and plots the fitted
        curve, but does not explicitly return them. The figure will be shown or
        available in the current matplotlib backend.
    """
    xFit = np.linspace(0, 2*np.pi, 4*36)
    yFit = fits.FitFunc.COS.call(coeffs, xFit)
    _, ax = plt.subplots(figsize=(8,6))
    ax.plot(xFit, yFit, color='red', label='fit')
    ax.set_xlabel("Dihedral angle / rad")
    ax.set_ylabel("Normalized count")
    if smarts:
        ax.set_title(f"{smarts}")
    return 

def PlotDihedralDistributions(m, dihedrals):
    cids = [x.GetId() for x in m.GetConformers()]
    dihedralDists = {}
    hists = {}
    for d in dihedrals:
        dihedralDists[d] = []
        hists[d] = []
    for i in range(len(cids)):
        for d in dihedrals:
            dihedralDists[d].append(rdMolTransforms.GetDihedralRad(m.GetConformer(cids[i]),d[0],d[1],d[2],d[3]))
    for i, d in enumerate(dihedrals):
        hists[d] = np.histogram(dihedralDists[d],bins=np.arange(-np.pi,np.pi,10*np.pi/180),density=False)

    if len(dihedrals) == 1:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        count = 0
        ax.bar(hists[dihedrals[count]][1][0:35],hists[dihedrals[count]][0],width=0.2,color='0.85',edgecolor='0.4')
        ax.set_title(f"{dihedrals[count]}")
        ax.set_xlabel("Dihedral angle / rad")
        ax.set_ylabel("Count")
    else:
        rows = int(len(dihedrals)/2 + len(dihedrals)%2)
        fig, ax = plt.subplots(rows,2,figsize=(10,rows*5))
        count = 0
        if rows > 1:
            for i in range(rows):
                for j in range(2):
                    if count < len(dihedrals):
                        ax[i][j].bar(hists[dihedrals[count]][1][0:35],hists[dihedrals[count]][0],width=0.2,color='0.85',edgecolor='0.4')
                        ax[i][j].set_title(f"{dihedrals[count]}")
                        ax[i][j].set_xlabel("Dihedral angle / rad")
                        ax[i][j].set_ylabel("Count")
                        count += 1
        else:
            for j in range(2):
                if count < len(dihedrals):
                    ax[j].bar(hists[dihedrals[count]][1][0:35],hists[dihedrals[count]][0],width=0.2,color='0.85',edgecolor='0.4')
                    ax[j].set_title(f"{dihedrals[count]}")
                    ax[j].set_xlabel("Dihedral angle / rad")
                    ax[j].set_ylabel("Count")
                    count += 1
    return fig

def VisualizeEnsemble(mol, dihedral=[], showTABS=False):    
    # build in the hoovering functionality:
    # when hovering over the atoms, the id should show
    colours=('cyanCarbon','redCarbon','blueCarbon','magentaCarbon','whiteCarbon','purpleCarbon')
    if mol.GetNumConformers() < 1:
        raise ValueError("No conformers in the molecule.")
    if showTABS:
        from tabs.torsions import DihedralInfoFromTorsionLib
        torInfo = DihedralInfoFromTorsionLib(mol)
        confTABS = torInfo.GetTABS()
    def DrawConformer(confId):
        if showTABS:
            print(f"TABS: {confTABS[confId]}")
        p = py3Dmol.view(width=400, height=400)
        p.removeAllModels()
        IPythonConsole.addMolToView(mol,p,confId=confId)
        p.setStyle({"stick": {}})
        # p.setStyle({'model':0,},
                            # {'stick':{'colorscheme':colours[0%len(colours)]}})
        if dihedral:
            for atomId in dihedral:
                p.setStyle({"serial": atomId}, {"stick": {"color": "red"}})
        p.zoomTo()
        return p.show()

    return interact(DrawConformer, confId=IntSlider(min=0, max=mol.GetNumConformers()-1, step=1, value=0))

def CpfPlot(populationMatrices, labels=None, title="", fontSize=15, markerSize=10):
    if isinstance(populationMatrices, np.ndarray):
        populationMatrices = [populationMatrices]

    plt.rcParams.update({'font.size': fontSize})
    # plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#14498e","#e06800"]
    markers = ['o', 'x']
    for idx, populationMatrix in enumerate(populationMatrices):
        final_pop = np.asarray(populationMatrix)[:, -1]
        sort_idx = np.argsort(final_pop)[::-1]
        sorted_final_pop = final_pop[sort_idx]
        cumulative_pop = np.cumsum(sorted_final_pop) / np.sum(sorted_final_pop)
        label = labels[idx] if labels is not None else (f'{idx+1}' if len(populationMatrices) > 1 else 'none')
        ax.scatter(np.arange(len(cumulative_pop)), cumulative_pop, s=markerSize, alpha=0.9, label=label, color=colors[idx], marker=markers[idx])

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("customTABS Index (sorted by final population)")
    ax.set_ylabel("CPF")
    ax.set_title(title, weight="bold")
    ax.grid(True)
    ax.legend()
    return fig

def PlotTop10Stackplot(popMatrix, timestep):
    """
    Plots a stackplot of the top 10 relative populations over time.

    Parameters:
        popMatrix (np.ndarray): Population matrix (states x frames)
        timestep (float): Time per frame in ns

    Returns:
        fig, ax: The matplotlib Figure and Axes objects.
    """
    indices = np.argsort(popMatrix[:,-1])[-10:][::-1]
    n_frames = popMatrix.shape[1]
    frames = np.arange(1, n_frames + 1)
    time = frames * timestep
    y = np.array([popMatrix[idx] / frames for idx in indices])

    fig, ax = plt.subplots(figsize=(10,6))
    ax.stackplot(time, y, labels=[f'State {idx}' for idx in indices], alpha=0.7)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Relative Population')
    ax.set_title('Top 10 Relative Populations Over Time (Stacked)')
    ax.legend(loc='upper left')
    return fig, ax
