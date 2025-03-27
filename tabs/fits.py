import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelextrema, find_peaks
import enum

# global vars
BINSIZE = None
FALLBACK_BINSIZE = 2*np.pi/36 # 10 degree bins
NSAMPLES = None

class FitFunc(enum.IntEnum):
    COS   = 1,
    GAUSS = 2,

    def call(self, params, x):
        period = 2*np.pi

        if self == FitFunc.COS:
            y = _ffitnew(x, *params)
            y /= max(y)
            return y

        elif self == FitFunc.GAUSS:
            params = np.reshape(params, (-1, 3))
            y = np.zeros(len(x))

            for p in params:
                a = p[0]
                b = p[1]
                c = p[2]

                diff = np.mod(x - b + period/2, period) - period/2
                y += a * np.exp(-((diff) / c)**2)
            return y

        else: return None

def _setBinsize(binsize):
    if binsize is not None:
        global BINSIZE
        BINSIZE = binsize
    else:
        global FALLBACK_BINSIZE
        BINSIZE = FALLBACK_BINSIZE

def _setNSamples(nSamples):
    global NSAMPLES
    NSAMPLES = nSamples

class Histogram:
    def __init__(self, histCount, histDensity, histSmoothed):
        self.histCount = histCount
        self.histDensity = histDensity
        self.histSmoothed = histSmoothed

class PeakInfo:
    def __init__(self, binIndex, binIndexMirror, binsAssigned, peakAt2Pi, peakAtPi, totalNumBins, meanInitial=0.0, sigmaInitial = 0.0, heightInitial = 0.0):
        self.binIndex = binIndex
        self.binIndexMirror = binIndexMirror
        self.binsAssigned = binsAssigned
        self.peakAt2Pi = peakAt2Pi
        self.peakAtPi = peakAtPi
        if self.peakAt2Pi:
            self.fittingPeak = True
        elif self.peakAtPi:
            self.fittingPeak = True
        elif self.binIndex < totalNumBins//2:
            self.fittingPeak = True
        else:
            self.fittingPeak = False
        self.sigmaInitial = sigmaInitial
        self.heightInitial = heightInitial
        self.meanInitial = meanInitial

def _WrappedGaussian(params, x, period=2*np.pi):
    """
    Simmilar to _Gaussian, but the result is periodic
    Parameters:
    - params (np.array(shape=(n, 3), dtype=float)): the wrapped gaussian parameters
    - x (List[float]): sample points
    - period (float, default=2*np.pi): wrap period
    """

    params = np.reshape(params, (-1, 3))
    y = np.zeros(len(x))
    
    for p in params:
        a = p[0]
        b = p[1]
        c = max(p[2], 1e-10)

        diff = np.mod(x - b + period/2, period) - period/2
        tmp = a * np.exp(-((diff) / c)**2)
        y += tmp

    return y

def findPeaksPeriodic(x, nPeaks, **kwargs):
    """ see: https://github.com/scipy/scipy/issues/13710 """

    splitindex = np.argmin(x)

    split1 = x[0:splitindex]
    split2 = x[splitindex:]

    shifted = np.concatenate((split2, split1))

    if "prominence" not in kwargs:
        kwargs["prominence"] = (None, None)

    peaks, info = find_peaks(shifted, **kwargs)
    if len(peaks) == 0: return peaks
    peaks = sorted(list(zip(peaks, info["prominences"])), key=lambda p: p[1], reverse=True)
    peaks, info = list(zip(*peaks))
    peaks = np.array(peaks[:nPeaks])
    info = info[:nPeaks]
    peaks = (peaks - len(split2)) % len(x)
    return peaks

def _PeaksAtLeastXDegreeApart(peaks):
    # what are 15 degrees in number of bins
    disTol = np.deg2rad(30)
    peaksAsAngles = [p*BINSIZE for p in peaks]
    disMat = np.zeros((len(peaks), len(peaks)))
    for i in range(len(peaks)):
        disMat[i,i] = 100
        for j in range(i+1,len(peaks)):
            dAngle = np.abs(peaksAsAngles[i] - peaksAsAngles[j])
            if dAngle > np.pi:
                dAngle = 2*np.pi - dAngle
            disMat[i,j] = dAngle
            disMat[j,i] = disMat[i,j]
    acceptedPeaks = []
    # only accept peak if it is at least 10 degrees apart from all other peaks
    for i in range(len(peaks)):
        if np.all(disMat[i,:i] > disTol):
            acceptedPeaks.append(peaks[i])

    return acceptedPeaks

def _PartitionPeaksSingleHist(hists, nMaxPeaks, peakThreshold=1e-2, excludePeaks=1e-2, promincence=1e-2):
    yHist = hists.histSmoothed
    yHistCount = hists.histCount
    peaksInitial = findPeaksPeriodic(yHist, nMaxPeaks, peakThreshold=peakThreshold, height=excludePeaks, prominence=promincence)
    peaksInitial = [int(p) for p in peaksInitial]
    peaksInitial = sorted(peaksInitial)
    peaks = []
    # only keep the peaks that are made up by at least 3 counts
    for p in peaksInitial:
        if np.sum(yHistCount[p-1:p+2]) > 3.0 :
            peaks.append(p)
        # else:
        #     print("peak at", p, "is excluded", np.sum(yHistCount[p-1:p+2]))
    peaks = _PeaksAtLeastXDegreeApart(peaks)    

def _PartitionPeaksHist(
        hists,
        nMaxPeaks=6,
        peakThreshold=1e-2,
        excludePeaks=1e-2,
        prominence=1e-2,
        ):
    """
    partition the histogram into multiple bins around the peaks

    Parameters:
    - nMaxPeaks         (int, default=4):           number of maximum peaks used to derive bins, if the number of peaks exceed it the lower peaks are excluded
    - peakThreshold    (float, default=1e-2):      fraction of the peak height to use for defining the peak width
    - smoothSD          (float, default=np.pi/4):   standard deviation for Gaussian smoothing kernel
    - excludePeaks      (float, default=1e-2):      peaks lower than this value will not be considered

    Returns:
    - List[int], List[int]: the indices of the bin borders and the peaks
    """
    binAngles = []
    binPeaks = []
    binIndxs, peaks, peaksInfo = _PartitionPeaksSingleHist(hists, nMaxPeaks, peakThreshold=peakThreshold, excludePeaks=excludePeaks, promincence=prominence)
    binPeaks.append(peaks)
    binAngles.append(binIndxs)
    
    return binAngles, binPeaks, peaksInfo

def _ConcatParams(arrList):
    params = np.hstack((tuple([np.reshape(a, (-1,1)) for a in arrList])))
    return params

def _BinFits(xHist, yHist, peaks, peaksInfo):
    peaks = peaks[0]
    
    def _ErrorWrapped(params, x, yRef, means):
        n = len(means)
        heights = params[:n]
        sigmas = params[n:]
        gaussParams = _ConcatParams([heights, means, sigmas])
        yFit = _WrappedGaussian(gaussParams, x)
        pointwiseError = np.mean(np.square(np.subtract(yFit, yRef)))
        return pointwiseError

    sigmas = []
    heights = []
    means = []

    for p in peaks:
        pi = peaksInfo[p]
        if not pi.fittingPeak:
            continue
        center = pi.meanInitial
        height = pi.heightInitial
        sigma = pi.sigmaInitial
        if pi.peakAtPi or pi.peakAt2Pi:
            height *= 0.5
        heights.append(height)
        means.append(center)
        sigmas.append(sigma)

    heightBounds = [(0.01, None) for height in heights]
    sigmaBounds = [(0.1, sigma+0.5) for sigma in sigmas]
    params = np.concatenate([heights, sigmas])
    bounds = heightBounds + sigmaBounds
    res = minimize(_ErrorWrapped, params, args=(xHist, yHist, means), bounds=bounds)
    heights = res.x[:len(means)]
    sigmas = res.x[len(means):]
    params = _ConcatParams([heights, means, sigmas])

    return params

def ComputeTorsionHistograms(dAngles, start, stop, step, density=True):
    if len(dAngles) == 0:
        return []
    
    hists=[]
    nDihedrals = len(dAngles[0])
    bins = np.arange(start,stop,step)
    
    edge = np.histogram(dAngles[:,0], bins=bins, density=density)[1]
    for i in range(nDihedrals):
        hist = np.histogram(dAngles[:,i], bins=bins, density=density)[0]
        hists.append(hist)

    return hists, edge[1:]

def ComputeGaussianFit(xHist, yHist, yHistCount, **kwargs):
    gaussBins, _ = _PartitionPeaksHist(yHist, **kwargs)
    coeffs = _BinFits(xHist, yHist, gaussBins)

    xFit = np.linspace(0, 2*np.pi, 2*len(xHist))
    yFit = FitFunc.GAUSS.call(coeffs, xFit)
    bins, _ = _PartitionPeaksHist(yFit, **kwargs)

    return coeffs, bins