import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.colors
import numpy as np

cmapSame = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","#fdb176"])
cmapDiff = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","#577d78"])

def PlotConfusionMatrixSchema():
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    confMatSchema = np.zeros((2,2))
    xlabelsconfMat = ['different','same']
    ylabelsconfMat = ['different','same']
    ax.imshow(confMatSchema,cmap='Blues')
    ax.set_xticks(np.arange(2), xlabelsconfMat,fontsize=24)
    ax.set_yticks(np.arange(2), ylabelsconfMat,fontsize=24)
    ax.xaxis.tick_top()
    ax.set_xlabel("TABS",fontsize=28)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("RMSD",fontsize=28)
    ax.text(0,0,"TN",fontsize=28)
    ax.text(0,1,"FN",fontsize=28)
    ax.text(1,0,"FP",fontsize=28)
    ax.text(1,1,"TP",fontsize=28)

    return fig

def PlotHist2dConfusionMatrices(same,different):
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    hDiff = ax[0].hist2d(different[:,0],different[:,1],bins=[1,30],cmap=cmapDiff,density=True,vmin=0,vmax=1)
    hSame = ax[1].hist2d(same[:,0],same[:,1],bins=[1,30],cmap=cmapSame,density=True,vmin=0,vmax=1)
    ax[0].set_ylim(0,6)
    ax[0].set_yticklabels(range(0,7),fontsize=22)
    ax[0].set_ylabel("RMSD / $\AA$",fontsize=22)
    ax[0].set_xticks([0.0])
    ax[0].set_xticklabels(["different TABS"],fontsize=28)
    cbar1 = fig.colorbar(hDiff[3], ax=ax[0],location="left",pad=0.35)
    cbar1.ax.yaxis.set_ticks_position('right')
    cbar1.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar1.ax.tick_params(labelsize=18)
    cbar1.ax.yaxis.get_offset_text().set_fontsize(18)
    cbar1.set_label('Density', fontsize=28)
    ax[1].set_ylim(0,6)
    ax[1].set_xticks([1.0])
    ax[1].set_xticklabels(["same TABS"],fontsize=28)
    ax[1].set_yticklabels([])
    cbar2 = fig.colorbar(hSame[3], ax=ax[1],location="right",pad=0.35)
    cbar2.ax.yaxis.set_ticks_position('left')
    cbar2.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar2.ax.tick_params(labelsize=18)
    cbar2.ax.yaxis.get_offset_text().set_fontsize(18)
    cbar2.set_label('Density', fontsize=28)

    return fig

def PlotPpvNpv(sumCms,npvs,ppvs,categoryForTitle):
    xlabels = list(sumCms.keys())
    ylabels = [0.0,0.2,0.4,0.6,0.8,1.0]
    fig, ax = plt.subplots(1,1,figsize=(15,10))
    ax.set_yticks(ylabels)
    ax.set_yticklabels(ylabels,fontsize=28)
    ax.set_ylabel("probability",fontsize=28)
    ax.set_xticks(xlabels)
    ax.set_xticklabels(xlabels,fontsize=20,rotation=45)
    ax.set_xlabel("RMSD threshold / $\AA$",fontsize=28)
    ax.plot(xlabels,ppvs,label="PPV",color="#fdb176",marker="o",markersize=10)
    ax.plot(xlabels,npvs,label="NPV",color="#577d78",marker="^",markersize=10)
    ax.legend(fontsize=28)
    ax.set_title(f"{categoryForTitle} flexibility category",fontsize=28)
    ax.vlines(0.9,0,1,linestyle="--",color="grey",linewidth=4)

    return fig

def PlotNTabsCorrelationAnalysis(dat):
    plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplot_mosaic([['A','B']],
                                layout="constrained",
                                figsize=(24,10)
                                )
    h = ax['A'].hist2d(np.log10(dat["ntabs"]),np.log10(dat["nconfsout"]),bins=(30,30),range=[[0,6],[0,6]],cmap='Blues',norm=matplotlib.colors.LogNorm(vmax=30))
    ax['A'].set_xlabel("log10(nTABS)")
    ax['A'].set_ylabel("log10(nConfsOut)")
    ax['A'].plot(np.linspace(0,6,6),np.linspace(0,6,6),color="red")
    # ax['A'].hlines(np.log10(5000),0,6,linestyles="--")
    ax['A'].hlines(np.log10(500000),0,6,linestyles="--")
    #ax['A'].plot(np.linspace(0,6,6),np.linspace(1,7,6),color="red")
    box = ax['A'].get_position()
    ax['A'].set_position([box.x0, box.y0, box.width, box.height*0.8])
    cbar = fig.colorbar(h[3], ticks=[1,5,10,20,30], ax=ax['A'])
    cbar.ax.set_yticklabels(['1','5','10','20','> 30'])
    cbar.set_label("count",fontsize=25,labelpad=0,rotation=270)
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax['A'].text(0.1, 0.75, "A)\n", transform=ax["A"].transAxes + trans,
                fontsize=35, va='bottom', weight='bold')

    ax['B'].hist(np.log10(dat["nconfsout"])-np.log10(dat["ntabs"]),bins=50,density=True)
    # ax['B'].vlines(0,ymin=0,ymax=1,color='r')
    ax['B'].vlines(np.median(np.log10(dat["nconfsout"])-np.log10(dat["ntabs"])),ymin=0,ymax=1,color='r')
    ax['B'].vlines(np.percentile(np.log10(dat["nconfsout"])-np.log10(dat["ntabs"]),25),0,1,color='r',linestyle="--")
    ax['B'].vlines(np.percentile(np.log10(dat["nconfsout"])-np.log10(dat["ntabs"]),75),0,1,color='r',linestyle="--")
    ax['B'].set_xlabel("$\Delta$(log10(nConfsOut),log10(nTABS))")
    ax['B'].set_ylabel("probability")
    box = ax['B'].get_position()
    ax['B'].set_position([box.x0, box.y0, box.width * 0.8, box.height*0.8])
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax['B'].text(0.1, 0.75, "B)\n", transform=ax["B"].transAxes + trans,
                fontsize=35, va='bottom',weight='bold')
    ax['B'].set_ylim(0,1)
    
    return fig