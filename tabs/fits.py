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
    Similar to a Gaussian function, but the result is periodic.

    Parameters
    ----------
    params : np.ndarray
        A 2D array of shape (n, 3) containing the wrapped Gaussian parameters.
        Each row corresponds to a set of parameters [a, b, c], where:
        - a (float): Amplitude of the Gaussian.
        - b (float): Center of the Gaussian.
        - c (float): Width of the Gaussian (must be greater than or equal to 1e-10).
    x : list of float
        Sample points at which the wrapped Gaussian is evaluated.
    period : float, optional
        The wrap period of the function. Default is 2 * π.
    Returns
    -------
    np.ndarray
        The periodic Gaussian function evaluated at the sample points `x`.
    Notes
    -----
    - The function ensures periodicity by wrapping the difference between
      `x` and the center `b` within the specified period.
    - The width parameter `c` is constrained to a minimum value of 1e-10
      to avoid numerical instability.
    """
    
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
    def __init__(self, histCount, histDensity, histSmoothed, binEdges=[]):
        self.histCount = histCount
        self.histDensity = histDensity
        self.histSmoothed = histSmoothed
        self.xHist = np.arange(GlobalVars.BINSIZE*0.5, 2*np.pi+0.5*GlobalVars.BINSIZE, GlobalVars.BINSIZE)

    def _PeakDetection(self,
                        nMaxPeaks=6,
                        excludePeaks=1e-2,
                        prominence=1e-3,
                        minCountsForPeak=3.0
                        ):
        yHist = self.histSmoothed
        yHistCount = self.histCount
        peaksInitial = _findPeaksPeriodic(yHist, nMaxPeaks, height=excludePeaks, prominence=prominence)
        peaksInitial = [int(p) for p in peaksInitial]
        peaksInitial = sorted(peaksInitial)
        peaks = []
        for p in peaksInitial:
            if np.sum(np.take(yHistCount,[p-1,p,p+1],mode="wrap")) > minCountsForPeak: 
                peaks.append(p)
        peaks = _PeaksAtLeastXDegreeApart(peaks)

        return peaks

    def _ComputeWeightedMean(self, indices):
        values = np.take(self.xHist,indices) # xHist
        weights = np.take(self.histDensity,indices) # yHist
        weights = weights - np.min(weights)
        mean = np.sum(values*weights)/np.sum(weights)
        return mean

    def _PartitionPeaksHist(self,
                            nMaxPeaks=6,
                            excludePeaks=1e-4,
                            prominence=1e-4,
                            mergePeaks=False,
                            ):
        """
        partition the histogram into multiple bins around the peaks

        Parameters:
        -----------
        - nMaxPeaks         (int, default=4):           number of maximum peaks used to derive bins, if the number of peaks exceed it the lower peaks are excluded
        - peakThreshold    (float, default=1e-2):      fraction of the peak height to use for defining the peak width
        - smoothSD          (float, default=np.pi/4):   standard deviation for Gaussian smoothing kernel
        - excludePeaks      (float, default=1e-2):      peaks lower than this value will not be considered

        Returns:
        --------
        - List[int], List[int]: the indices of the bin borders and the peaks
        """
        
        yHist = self.histSmoothed

        peaks = self._PeakDetection(nMaxPeaks=nMaxPeaks, excludePeaks=excludePeaks, prominence=prominence)
       
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
            tmp = _FindValley(lpeak, rpeak, yHist)
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
                    # don't have to do the second set here; can take sequence etc as the second argument
                    # maybe issue with range object?
                    binIndices = set(range(leftEdge,middle)).union(set(range(middle,middle+(middle-leftEdge)))) 
                elif peakAt2Pi:
                    index = len(yHist) - 1
                    meanGuess = 2*np.pi
                    binIndices = set(range(len(yHist)-min(binEdges),len(yHist))).union(set(range(0,min(binEdges))))
                else:
                    index = peak
                    meanGuess = self.xHist[peak]
                peaksInfo[peak] = PeakInfo(binIndex=index, 
                                           binIndexMirror=-1, 
                                           binsAssigned=binIndices, 
                                           peakAt2Pi=peakAt2Pi, 
                                           peakAtPi=peakAtPi, 
                                           leftEdge=leftEdge, 
                                           rightEdge=rightEdge, 
                                           meanInitial=meanGuess)

            for i, peak in enumerate(peaks):
                leftEdge = binEdges[i-1]
                pi = peaksInfo[peak]
                pi.sigmaInitial = _EstimateSigma(self.histSmoothed[peak], peak, self.histSmoothed[leftEdge], leftEdge, self.histSmoothed)
                pi.heightInitial = self.histDensity[peak]
        else:
            peak = peaks[0]
            peakAtPi = IsPeakNearPi(peak, eps)
            peakAt2Pi = IsPeakNear2Pi(peak, eps)
            leftEdge = binEdges[-1]
            rightEdge = binEdges[0]
            if peakAtPi:
                index = middle
                meanGuess = np.pi
            elif peakAt2Pi:
                index = len(yHist) - 1
                meanGuess = 2*np.pi
            else:
                index = peak
                meanGuess = self.xHist[peak]
            pi = PeakInfo(binIndex=index, 
                          binIndexMirror=-1, 
                          binsAssigned=set(range(len(yHist))), 
                          peakAt2Pi=peakAt2Pi, 
                          peakAtPi=peakAtPi, 
                          leftEdge=leftEdge, 
                          rightEdge=rightEdge, 
                          meanInitial=meanGuess)
            pi.sigmaInitial = _EstimateSigma(self.histSmoothed[peak], peak, self.histSmoothed[binEdges[0]], binEdges[0], self.histSmoothed)
            pi.heightInitial = self.histDensity[peak]
            peaksInfo[peak] = pi

        for pKey in peaksInfo:
            pi = peaksInfo[pKey]
            # print(pi.binIndex*GlobalVars.BINSIZE, pi.binsAssigned, pi.meanInitial, pi.sigmaInitial, pi.heightInitial)

        if len(peaks)> 1 and mergePeaks:
            peaksNew = deepcopy(peaks)
            peaksInfoNew = deepcopy(peaksInfo)
            keys = list(peaksInfo.keys())
            for i, key1 in enumerate(keys):
                for j, key2 in enumerate(keys[i+1:]):
                    dis = np.abs(peaksInfo[key1].meanInitial - peaksInfo[key2].meanInitial)
                    # initial guess for left and right
                    leftPeak = key1
                    rightPeak = key2
                    if dis > np.pi:
                        dis = 2 * np.pi - dis
                        leftPeak = key2
                        rightPeak = key1
                    if dis < 0.7:
                        pi = PeakInfo(binIndex=0.5*(peaksInfo[leftPeak].rightEdge+peaksInfo[rightPeak].leftEdge), 
                                      binIndexMirror=-1, 
                                      binsAssigned=peaksInfo[leftPeak].binsAssigned.union(peaksInfo[rightPeak].binsAssigned), 
                                      peakAtPi=False, 
                                      peakAt2Pi=False, 
                                      leftEdge=peaksInfo[leftPeak].leftEdge, 
                                      rightEdge=peaksInfo[rightPeak].rightEdge, 
                                      meanInitial=self.xHist[peaksInfo[leftPeak].rightEdge])
                        pi.sigmaInitial = _EstimateSigma(self.histSmoothed[peak], peak, self.histSmoothed[leftEdge], leftEdge, self.histSmoothed)
                        pi.heightInitial = self.histDensity[peak]
                        peaksInfoNew[peaksInfo[leftPeak].rightEdge] = pi
                        del peaksInfoNew[key1]
                        del peaksInfoNew[key2]
                        peaksNew.remove(key1)
                        peaksNew.remove(key2)
                        peaksNew.append(peaksInfo[leftPeak].rightEdge)
            peaks = peaksNew
            peaksInfo = peaksInfoNew            
        return binEdges, peaks, peaksInfo


class PeakInfo:
    def __init__(self, binIndex, binIndexMirror, binsAssigned, peakAt2Pi, peakAtPi, leftEdge, rightEdge, meanInitial=0.0, sigmaInitial = 0.0, heightInitial = 0.0):
        self.binIndex = binIndex
        self.binIndexMirror = binIndexMirror
        self.binsAssigned = binsAssigned
        self.peakAt2Pi = peakAt2Pi
        self.peakAtPi = peakAtPi
        self.fittingPeak = True
        self.sigmaInitial = sigmaInitial
        self.heightInitial = heightInitial
        self.meanInitial = meanInitial
        self.leftEdge = leftEdge
        self.rightEdge = rightEdge


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

def _FindValley(p1, p2, yHist):
    if p2 < p1:
        p2 = len(yHist) + p2
    reducedHist = np.take(yHist, range(p1, p2), mode="wrap")
    a = 0
    b = len(reducedHist)-1
    while reducedHist[a+1]-reducedHist[a] <= 0:
        a += 1
        if a == len(reducedHist)-1:
            break
    while reducedHist[b]-reducedHist[b-1] >= 0:
        b -= 1
        if b == 1:
            break
    valley = int((a+b)*0.5)
    valleyTranslated = (p1 + valley)%len(yHist)
    return valleyTranslated

def _EstimateSigma(peak,peakPos,tail,tailPos,yHist):
    if tail > peak:
        peak += 2*np.pi
    fraction = max(yHist[tailPos]/yHist[peakPos],1e-12)
    # actually this is not ideal
    fraction = min(fraction,0.99)
    delta_x = np.abs(peak - tail)
    if delta_x > np.pi:
        delta_x = 2*np.pi - delta_x
    sigma = np.sqrt(-(delta_x ** 2) / (2 * np.log(fraction)))
    return sigma

def _ConcatParams(arrList):
    params = np.hstack((tuple([np.reshape(a, (-1,1)) for a in arrList])))
    return params

def _ErrorSigma(sigmas, x, yRef, heights, means):
    gaussParams = _ConcatParams([heights, means, sigmas])
    yFit = _WrappedGaussian(gaussParams, x)
    return np.sum(np.square(np.subtract(yFit, yRef)))

def _ErrorHeight(heights, x, yRef, means, sigmas):
    gaussParams = _ConcatParams([heights, means, sigmas])
    yFit = _WrappedGaussian(gaussParams, x)
    return np.sum(np.square(np.subtract(yFit, yRef)))

def _BinFits(xHist, yHist, peaks, peaksInfo):
    sigmas = []
    heights = []
    means = []

    for p in peaks:
        peakInfo = peaksInfo[p]
        if not peakInfo.fittingPeak:
            continue
        heights.append(peakInfo.heightInitial)
        means.append(peakInfo.meanInitial)
        sigmas.append(peakInfo.sigmaInitial)

    heightBounds = [(height-0.01, height+0.01) for height in heights]
    sigmaBounds = [(0.1, sigma+0.7) for sigma in sigmas]
    res = minimize(_ErrorSigma, sigmas, args=(xHist, yHist, heights, means), bounds=sigmaBounds)
    sigmas = res.x
    res = minimize(_ErrorHeight, heights, args=(xHist, yHist, means, sigmas), bounds=heightBounds)
    heights = res.x
    params = _ConcatParams([heights, means, sigmas])

    return params

def _BoundDetectionOnFit(xFit, yFit, coeffs):
    peaks = coeffs[:,1]
    #translate the peaks into their idx
    peaks = np.array([int(np.round(p/xFit[1])) for p in peaks])
    peakPairs = []
    for i in range(len(peaks)-1):
        peakPairs.append((peaks[i], peaks[i+1]))
    if len(peaks) > 1:
        peakPairs.append((peaks[-1], peaks[0]))
    if not peakPairs:
        peakPairs = [(peaks[0], peaks[0])]

    bounds = []
    for lpeak, rpeak in peakPairs:
        if lpeak == rpeak:
            # edge should be half the length of the histogram away from the peak
            border = lpeak-len(yFit)//2
            if border < 0:
                border = len(yFit) + border
            bounds.append(xFit[border])
            continue
        tmp = _FindValley(lpeak, rpeak, yFit)
        bounds.append(xFit[tmp])

    return bounds

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

def ComputeGaussianFit(xHist, yHist, yHistCount, binsize, sigma=np.pi*0.3, **kwargs):
    GlobalVars.SetBinsize(binsize)
    GlobalVars.SetNSamples(np.sum(yHistCount))
    # make sigma a keyword argument
    yHistSmooth = gaussian_filter1d(yHist, sigma=sigma, mode="wrap")
    yHistCount = gaussian_filter1d(yHistCount, sigma=sigma, mode="wrap")
    hists = Histogram(yHistCount, yHist, yHistSmooth)
    _, peaks, peaksInfo = hists._PartitionPeaksHist(**kwargs)
    coeffs = _BinFits(xHist, yHistSmooth, peaks, peaksInfo)
    xFit = np.linspace(0, 2*np.pi, 2*len(xHist))
    yFit = FitFunc.GAUSS.call(coeffs, xFit)
    bounds = _BoundDetectionOnFit(xFit, yFit, coeffs)

    return coeffs, bounds