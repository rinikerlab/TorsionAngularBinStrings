import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelextrema, find_peaks
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy
import enum

class GlobalVars:
    BINSIZE = None
    FALLBACK_BINSIZE = 2*np.pi/36 # 10 degree bins
    NSAMPLES = None

    @classmethod
    def SetBinsize(cls, binsize):
        cls.BINSIZE = binsize if binsize is not None else cls.FALLBACK_BINSIZE
    @classmethod
    def SetNSamples(cls, nSamples):
        cls.NSAMPLES = nSamples

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

def _ffitnew(x, s1, v1, s2, v2, s3, v3, s4, v4, s5, v5, s6, v6):
    # fitting as in Sereinas code
    c = np.cos(x)
    c2 = c*c
    c4 = c2*c2
    return np.exp(-(v1*(1+s1*c) + v2*(1+s2*(2*c2-1)) + v3*(1+s3*(4*c*c2-3*c)) \
                    + v4*(1+s4*(8*c4-8*c2+1)) + v5*(1+s5*(16*c4*c-20*c2*c+5*c)) \
                    + v6*(1+s6*(32*c4*c2-48*c4+18*c2+1)) ))

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

class Histogram:
    def __init__(self, histCount, histDensity, histSmoothed):
        self.histCount = histCount
        self.histDensity = histDensity
        self.histSmoothed = histSmoothed

    def _PartitionPeaksHist(self,
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
        
        yHist = self.histSmoothed
        yHistCount = self.histCount
        peaksInitial = _findPeaksPeriodic(yHist, nMaxPeaks, height=excludePeaks, prominence=prominence)
        peaksInitial = [int(p) for p in peaksInitial]
        peaksInitial = sorted(peaksInitial)
        peaks = []
        # only keep the peaks that are made up by at least 3 counts
        for p in peaksInitial:
            if np.sum(np.take(yHistCount,[p-1,p,p+1],mode="wrap")) > 3.0 :
                peaks.append(p)
            # else:
            #     print("peak at", p, "is excluded", np.sum(yHistCount[p-1:p+2]))
        peaks = _PeaksAtLeastXDegreeApart(peaks)
        peakPairs = []
        for i in range(len(peaks)-1):
            peakPairs.append((peaks[i], peaks[i+1]))
        if len(peaks) > 1:
            peakPairs.append((peaks[-1], peaks[0]))
        if not peakPairs:
            peakPairs = [(peaks[0], peaks[0])]
        binEdges = []
        for lpeak, rpeak in peakPairs:
            if lpeak == rpeak:
                # edge should be half the length of the histogram away from the peak
                border = lpeak-len(yHist)//2
                if border < 0:
                    border = len(yHist) + border
                binEdges.append(border)
                continue
            tmp = _FindValleyInvertedHist(lpeak, rpeak, yHist)
            binEdges.append(tmp)
        middle = len(yHist) // 2
        # already detect all the peak information here and not in BinFit
        # binEdges are the right edges 
        IsPeakNearPi = lambda peak, epsilon: abs(peak - len(yHist)//2) < epsilon 
        IsPeakNear2Pi = lambda peak, epsilon: abs(peak - len(yHist)) < epsilon or (abs(peak)) < epsilon 
        peaksInfo = {}
        tol = 0.35
        eps = tol//GlobalVars.BINSIZE
        if len(peaks) > 1:
            for i, peak in enumerate(peaks):
                peak = int(peak)
                peakAtPi = IsPeakNearPi(peak, eps)
                peakAt2Pi = IsPeakNear2Pi(peak, eps)
                leftEdge = binEdges[i-1]
                rightEdge = binEdges[i]
                if leftEdge < rightEdge:
                    binIndices = set(range(leftEdge,rightEdge))
                else:
                    binIndices = set(range(leftEdge,len(yHist))).union(set(range(0,rightEdge)))
                if peakAtPi:
                    index = middle
                    meanGuess = np.pi
                    binIndices = set(range(leftEdge,middle)).union(set(range(middle,middle+(middle-leftEdge)))) 
                elif peakAt2Pi:
                    index = len(yHist) - 1
                    meanGuess = 2*np.pi
                    binIndices = set(range(len(yHist)-min(binEdges),len(yHist))).union(set(range(0,min(binEdges))))
                else:
                    if rightEdge <= leftEdge: rightEdge += len(yHist)
                    sub = np.take(yHist, range(leftEdge, rightEdge), mode="wrap")
                    xHist = [i*GlobalVars.BINSIZE+0.5*GlobalVars.BINSIZE for i in range(len(yHist))]
                    xsub = np.take(xHist, range(leftEdge, rightEdge), mode="wrap")
                    index = peak
                    meanGuess = (_WeightedMean(xsub, sub)+xHist[peak])*0.5

                peaksInfo[peak] = PeakInfo(index, -1, binIndices, peakAt2Pi, peakAtPi, len(yHist), meanGuess)

            for i, peak in enumerate(peaks):
                leftEdge = binEdges[i-1]
                pi = peaksInfo[peak]
                pi.sigmaInitial = _EstimateSigma(peak*GlobalVars.BINSIZE+0.5*GlobalVars.BINSIZE, peak, leftEdge*GlobalVars.BINSIZE+0.5*GlobalVars.BINSIZE, leftEdge, yHist)
                pi.heightInitial = yHist[peak]
        else:
            peak = peaks[0]
            peakAtPi = IsPeakNearPi(peak, eps)
            peakAt2Pi = IsPeakNear2Pi(peak, eps)
            if peakAtPi:
                index = middle
                meanGuess = np.pi
            elif peakAt2Pi:
                index = len(yHist) - 1
                meanGuess = 2*np.pi
            else:
                index = peak
                leftEdge = binEdges[-1]
                rightEdge = binEdges[0]
                if rightEdge <= leftEdge: rightEdge += len(yHist)
                sub = np.take(yHist, range(leftEdge, rightEdge), mode="wrap")
                xHist = [i*GlobalVars.BINSIZE+0.5*GlobalVars.BINSIZE for i in range(len(yHist))]
                xsub = np.take(xHist, range(leftEdge, rightEdge), mode="wrap")
                meanGuess = (_WeightedMean(xsub, sub)+xHist[peak])*0.5
            pi = PeakInfo(index, -1, set(range(len(yHist))), peakAt2Pi, peakAtPi, len(yHist), meanGuess)
            pi.sigmaInitial = _EstimateSigma(peak*GlobalVars.BINSIZE+0.5*GlobalVars.BINSIZE, peak, binEdges[0]*GlobalVars.BINSIZE+0.5*GlobalVars.BINSIZE, binEdges[0], yHist)
            pi.heightInitial = yHist[peak]
            peaksInfo[peak] = pi

        return binEdges, peaks, peaksInfo


class PeakInfo:
    def __init__(self, binIndex, binIndexMirror, binsAssigned, peakAt2Pi, peakAtPi, totalNumBins, meanInitial=0.0, sigmaInitial = 0.0, heightInitial = 0.0):
        self.binIndex = binIndex
        self.binIndexMirror = binIndexMirror
        self.binsAssigned = binsAssigned
        self.peakAt2Pi = peakAt2Pi
        self.peakAtPi = peakAtPi
        self.fittingPeak = True
        self.sigmaInitial = sigmaInitial
        self.heightInitial = heightInitial
        self.meanInitial = meanInitial



def _findPeaksPeriodic(x, nPeaks, **kwargs):
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
    peaksAsAngles = [p*GlobalVars.BINSIZE for p in peaks]
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

def _FindValleyInvertedHist(p1, p2, yHist):
    p2Org = deepcopy(p2)
    if p2 < p1:
        p2 = len(yHist) + p2
    reducedHist = np.take(yHist, range(p1, p2), mode="wrap")
    invertedHist = 1 - reducedHist
    invertedHist = gaussian_filter1d(invertedHist, sigma=1.5)
    for i, val in enumerate(invertedHist):
        if i < len(invertedHist)-1:
            delta = invertedHist[i+1] - val
            if delta < 0:
                break
    tmpValley = deepcopy(i)
    for i, val in enumerate(reversed(invertedHist)):
        if i < len(invertedHist)-1:
            delta = invertedHist[i+1] - val
            if delta < 0:
                break
    tmpValley2 = deepcopy(len(invertedHist)-i)
    valley = int((tmpValley+tmpValley2)*0.5)
    valleyTranslated = 0
    if p1 < p2Org:
        valleyTranslated = p1+valley
    else:
        valleyTranslated = p1+valley-len(yHist)

    return valleyTranslated

def _EstimateSigma(peak,peakPos,tail,tailPos,yHist):
    if tail > peak:
        peak += 2*np.pi
    # print(tailPos, yHist[tailPos], peakPos, yHist[peakPos])
    fraction = max(yHist[tailPos]/yHist[peakPos],1e-12)
    # actually this is not ideal
    fraction = min(fraction,0.99)
    delta_x = np.abs(peak - tail)
    # print(delta_x)
    if delta_x > np.pi:
        delta_x = 2*np.pi - delta_x
    # print(delta_x, fraction)
    sigma = np.sqrt(-(delta_x ** 2) / (2 * np.log(fraction)))
    return sigma

def _WeightedMean(xSub, ySub):
    return np.sum(xSub * ySub) / np.sum(ySub)

def _ConcatParams(arrList):
    params = np.hstack((tuple([np.reshape(a, (-1,1)) for a in arrList])))
    return params

def _BinFits(xHist, yHist, peaks, peaksInfo):
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

# def _BoundDetectionOnFit(xFit, yFit, peaks):
#     # define interval (left peak, right peak)
#     intervals = []
#     bounds = []
#     peaks = sorted(peaks)
#     if len(peaks) < 2:
#         inverted = []
#     return 

def ComputeTorsionHistograms(customTorsionProfiles, binsize):
    if customTorsionProfiles is None:
        return []
    hists=[]
    histsCount=[]
    nDihedrals = customTorsionProfiles.shape[1]
    bins = np.arange(0, 2*np.pi+binsize, binsize)
    for i in range(nDihedrals):
        hist, xhist = np.histogram(customTorsionProfiles[:,i], bins=bins, density=True)
        histCount, _ = np.histogram(customTorsionProfiles[:,i], bins=bins, density=False)
        hists.append(hist)
        histsCount.append(histCount)

    return hists, histsCount, xhist[:-1]+0.5*binsize

def ComputeGaussianFit(xHist, yHist, yHistCount, binsize, **kwargs):
    GlobalVars.SetBinsize(binsize)
    GlobalVars.SetNSamples(np.sum(yHistCount))
    yHistSmooth = gaussian_filter1d(yHist, sigma=np.pi*0.35, mode="wrap")
    yHistCount = gaussian_filter1d(yHistCount, sigma=np.pi*0.35, mode="wrap")
    hists = Histogram(yHistCount, yHist, yHistSmooth)
    _, peaks, peaksInfo = hists._PartitionPeaksHist(**kwargs)
    # _,  peaks, peaksInfo = _PartitionPeaksHist(hists, **kwargs)
    print(peaks)
    coeffs = _BinFits(xHist, yHistSmooth, peaks, peaksInfo)

    xFit = np.linspace(0, 2*np.pi, 2*len(xHist))
    yFit = FitFunc.GAUSS.call(coeffs, xFit)
    # bins, _ = _PartitionPeaksHist(yFit, **kwargs)
    bins = []

    return coeffs, bins