import matplotlib.pyplot as plt
import pickle
import numpy as np
from importlib.resources import files
import json
import math
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw import IPythonConsole
import importlib.util
import warnings
# import py3Dmol
from ipywidgets import interact, IntSlider

IPythonConsole.ipython_3d = True
plt.rcParams.update({'font.size': 12})

def CheckPackageInstalled(package_name):
    if importlib.util.find_spec(package_name) is None:
        warnings.warn(
            f"The package '{package_name}' is not installed. Please install it using conda.",
            UserWarning
        )

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

        
    rows = (nPlots + plotsPerRow - 1) // plotsPerRow
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
        plt.subplots_adjust(top=0.9)
    return fig

def _ffitnew(x, s1, v1, s2, v2, s3, v3, s4, v4, s5, v5, s6, v6):
    c = np.cos(x)
    c2 = c*c
    c4 = c2*c2

    return math.exp(-(v1*(1+s1*c) + v2*(1+s2*(2*c2-1)) + v3*(1+s3*(4*c*c2-3*c)) \
                    + v4*(1+s4*(8*c4-8*c2+1)) + v5*(1+s5*(16*c4*c-20*c2*c+5*c)) \
                    + v6*(1+s6*(32*c4*c2-48*c4+18*c2+1)) ))

def PlotOrgDistribution(smarts,patterntype):
    if patterntype == "r":
        pstats = pickle.load(open(str(files("tabs").joinpath('torsionPreferences/ETKDGv3Data',"nonringbonds_torsion_ana_all.pkl")),"rb"),encoding="latin1")
        with open(str(files("tabs").joinpath('torsionPreferences','torsionPreferences_v2_formatted.txt'))) as f: torsionPreferencesv2 = f.read()
    elif patterntype == "sr":
        pstats = pickle.load(open(str(files("tabs").joinpath('torsionPreferences/ETKDGv3Data',"ringbonds_torsion_ana_all_newpatterns.pkl")),"rb"),encoding="latin1")
        with open(str(files("tabs").joinpath('torsionPreferences','torsionPreferences_smallrings_formatted.txt'))) as f: torsionPreferencesv2 = f.read()
    tpv2 = dict(json.loads(torsionPreferencesv2))
    tmp = tpv2[smarts]
    tmp = tmp.split(" ")
    tmp_2 = []
    for i in range(12):
        tmp_2.append(tmp[i])
    y = [_ffitnew(j/180.0*np.pi, float(tmp_2[0]),float(tmp_2[1]),float(tmp_2[2]),float(tmp_2[3]),float(tmp_2[4]),float(tmp_2[5]),float(tmp_2[6]),float(tmp_2[7]),float(tmp_2[8]),float(tmp_2[9]),float(tmp_2[10]),float(tmp_2[11])) for j in range(0, 360, 1)]
    ## this is only for visualization purposes; normally, the fit should be already in the correct order of magnitude!!
    my = max(y)
    y = [j/my for j in y]
    xh = [(i*10) for i in range(0,36)]
    thisto = {}
    v = []
    for j in pstats[smarts]:
        if j < 0: v.append(j+360)
        else: v.append(j)
    h = np.histogram(v, bins=range(0, 370, 10), range=(0,360.))
    mh = float(max(h[0]))
    tyh = [j/mh for j in h[0]]
    thisto[smarts] = tyh
    _, ax = plt.subplots(figsize=(8,6))
    ax.bar(xh, thisto[smarts], width=10.0, color='0.85',edgecolor='0.4',zorder=2)
    ax.plot(range(0, 360, 1), y, 'r', lw=2)
    ax.set_ylim(0,1.0)
    ax.set_xlabel("Dihedral angle / °")
    ax.set_ylabel("Normalized count")
    ax.set_title(f"{smarts}")
    return 

def PlotOrgDistributionFitOnly(smarts,patterntype):
    if patterntype == "r":
        with open(str(files("tabs").joinpath('torsionPreferences','torsionPreferences_v2_formatted.txt'))) as f: torsionPreferencesv2 = f.read()
    elif patterntype == "sr":
        with open(str(files("tabs").joinpath('torsionPreferences','torsionPreferences_smallrings_formatted.txt'))) as f: torsionPreferencesv2 = f.read()
    elif patterntype == "m":
        with open(str(files("tabs").joinpath('torsionPreferences','torsionPreferences_macrocycles_formatted.txt'))) as f: torsionPreferencesv2 = f.read()
    tpv2 = dict(json.loads(torsionPreferencesv2))
    tmp = tpv2[smarts]
    tmp = tmp.split(" ")
    tmp_2 = []
    for i in range(12):
        tmp_2.append(tmp[i])
    y = [_ffitnew(j/180.0*np.pi, float(tmp_2[0]),float(tmp_2[1]),float(tmp_2[2]),float(tmp_2[3]),float(tmp_2[4]),float(tmp_2[5]),float(tmp_2[6]),float(tmp_2[7]),float(tmp_2[8]),float(tmp_2[9]),float(tmp_2[10]),float(tmp_2[11])) for j in range(0, 360, 1)]
    ## this is only for visualization purposes; normally, the fit should be already in the correct order of magnitude!!
    my = max(y)
    y = [j/my for j in y]
    _, ax = plt.subplots(figsize=(8,6))
    ax.plot(range(0, 360, 1), y, 'r', lw=2)
    ax.set_ylim(0,1.0)
    ax.set_xlabel("Dihedral angle / °")
    ax.set_ylabel("Normalized count")
    ax.set_title(f"{smarts}")
    return

def GetOrgDistribution(smarts,patterntype):
    if patterntype == "r":
        pstats = pickle.load(open(str(files("tabs").joinpath('torsionPreferences/ETKDGv3Data',"nonringbonds_torsion_ana_all.pkl")),"rb"),encoding="latin1")
        with open(str(files("tabs").joinpath('torsionPreferences','torsionPreferences_v2_formatted.txt'))) as f: torsionPreferencesv2 = f.read()
    elif patterntype == "sr":
        pstats = pickle.load(open(str(files("tabs").joinpath('torsionPreferences/ETKDGv3Data',"ringbonds_torsion_ana_all_newpatterns.pkl")),"rb"),encoding="latin1")
        with open(str(files("tabs").joinpath('torsionPreferences','torsionPreferences_smallrings_formatted.txt'))) as f: torsionPreferencesv2 = f.read()
    tpv2 = dict(json.loads(torsionPreferencesv2))
    v = []
    for j in pstats[smarts]:
        if j < 0: v.append(j+360)
        else: v.append(j)
    h = np.histogram(v, bins=range(0, 370, 10), range=(0,360.))
    return v, h

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
    with warnings.catch_warnings(record=True) as checker:
        warnings.simplefilter("always")
        CheckPackageInstalled('py3Dmol')
    if checker:
        for warning in checker:
            print(f"Warning caught:\n {warning.message}")
        return
    else:
        import py3Dmol
    # build in the hoovering functionality:
    # when hovering over the atoms, the id should show
    colours=('cyanCarbon','redCarbon','blueCarbon','magentaCarbon','whiteCarbon','purpleCarbon')
    if mol.GetNumConformers() < 1:
        raise ValueError("No conformers in the molecule.")
    if showTABS:
        from tabs.torsions import DihedralsInfo
        torInfo = DihedralsInfo.FromTorsionLib(mol)
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