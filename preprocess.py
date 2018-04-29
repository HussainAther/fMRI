import distutils.version

import numpy as np
from scipy import signal, stats, linalg
from sklearn.utils import gen_even_slices

np_version = distutils.version.LooseVersion(np.version.short_version).version

def _standard(signals, detrend=False, normalize=True):
    """ Center and norm a given signal (time is along first axis)
    Parameters
    ==========
    signals: numpy.ndarray
        Timeseries to standardize
    detrend: bool
        if detrending of timeseries is requested
    normalize: bool
        if True, shift timeseries to zero mean value and scale
        to unit energy (sum of squares).
    Returns
    =======
    std_signals: numpy.ndarray
        copy of signals, normalized.
    """
    if detrend:
        signals = signals.copy()
        signals -= np.mean(signals, axis=0)
        # Keeping "signals" dtype avoids some type conversion further down,
        # and can save a lot of memory if dtype is single-precision.
        reg = np.arange(signals.shape[0], dtype=signals.dtype)
        reg -= reg.mean()
        reg /= np.sqrt((reg ** 2).sum())
        reg = reg[:, np.newaxis]

        # No batching for small arrays
        if signals.shape[1] < 500:
            n_batches = 1
        else:
            n_batches = 10
        # This is fastest for C order.
        for batch in gen_even_slices(signals.shape[1], 10):
            signals[:, batch] -= np.dot(reg[:, 0], signals[:, batch]
                                        ) * reg
    
    else:
        signals = signals.copy()

    if normalize:
        # remove mean if not already detrended
        if not detrend:
            signals -= signals.mean(axis=0)

        std = np.sqrt((signals ** 2).sum(axis=0))
        std[std < np.finfo(np.float).eps] = 1.  # avoid numerical problems
        signals /= std
    return signals


def clean(signals, detrend=True, standardize=True, confounds=None,
          low_pass=None, high_pass=None, t_r=2.5):
    """Improve SNR on masked fMRI signals.

       This function can do several things on the input signals, in
       the following order:
       - detrend
       - standardize
       - remove confounds
       - low- and high-pass filter

       Low-pass filtering improves specificity.

       High-pass filtering should be kept small, to keep some
       sensitivity.

       Filtering is only meaningful on evenly-sampled signals.

       Parameters
       ==========
       signals: numpy.ndarray
           Timeseries. Must have shape (instant number, features number).
           This array is not modified.

       confounds: numpy.ndarray, str or list of
           Confounds timeseries. Shape must be
           (instant number, confound number), or just (instant number,)
           The number of time instants in signals and confounds must be
           identical (i.e. signals.shape[0] == confounds.shape[0]).
           If a string is provided, it is assumed to be the name of a csv file
           containing signals as columns, with an optional one-line header.
           If a list is provided, all confounds are removed from the input
           signal, as if all were in the same array.

       t_r: float
           Repetition time, in second (sampling period).

       low_pass, high_pass: float
           Respectively low and high cutoff frequencies, in Hertz.

       detrend: bool
           If detrending should be applied on timeseries (before
           confound removal)

       standardize: bool
           If True, returned signals are set to unit variance.

       Returns
       =======
       cleaned_signals: numpy.ndarray
           Input signals, cleaned. Same shape as `signals`.

       Notes
       =====
       Confounds removal is based on a projection on the orthogonal
       of the signal space. See `Friston, K. J., A. P. Holmes,
       K. J. Worsley, J.-P. Poline, C. D. Frith, et R. S. J. Frackowiak.
       "Statistical Parametric Maps in Functional Imaging: A General
       Linear Approach". Human Brain Mapping 2, no 4 (1994): 189-210.
       <http://dx.doi.org/10.1002/hbm.460020402>`_
    """

    if not isinstance(confounds,
                      (list, tuple, str, np.ndarray, type(None))):
        raise TypeError("confounds keyword has an unhandled type: %s"
                        % confounds.__class__)

    # Standardize / detrend
    normalize = False
    if confounds is not None:
        # If confounds are to be removed, then force normalization to improve
        # matrix conditioning.
        normalize = True
    signals = _standard(signals, normalize=normalize, detrend=detrend)

    # Remove confounds
    if confounds is not None:
        if not isinstance(confounds, (list, tuple)):
            confounds = (confounds, )

        # Read confounds
        all_confounds = []
        for confound in confounds:
            if isinstance(confound, str):
                filename = confound
                confound = np.genfromtxt(filename)
                if np.isnan(confound.flat[0]):
                    # There may be a header
                    if np_version >= [1, 4, 0]:
                        confound = np.genfromtxt(filename, skip_header=1)
                    else:
                        confound = np.genfromtxt(filename, skiprows=1)
                if confound.shape[0] != signals.shape[0]:
                    raise ValueError("Confound signal has an incorrect length")

            elif isinstance(confound, np.ndarray):
                if confound.ndim == 1:
                    confound = np.atleast_2d(confound).T
                elif confound.ndim != 2:
                    raise ValueError("confound array has an incorrect number "
                                     "of dimensions: %d" % confound.ndim)

                if confound.shape[0] != signals.shape[0]:
                    raise ValueError("Confound signal has an incorrect length")
            else:
                raise TypeError("confound has an unhandled type: %s"
                                % confound.__class__)
            all_confounds.append(confound)

        # Restrict the signal to the orthogonal of the confounds
        confounds = np.hstack(all_confounds)
        del all_confounds
        confounds = _standard(confounds, normalize=True, detrend=detrend)
        Q = qr_economic(confounds)[0]
        signals -= np.dot(Q, np.dot(Q.T, signals))

    if low_pass is not None or high_pass is not None:
        signals = butterworth(signals, sampling_rate=1. / t_r,
                              low_pass=low_pass, high_pass=high_pass)

    if standardize:
        signals = _standard(signals, normalize=True, detrend=False)
        signals *= np.sqrt(signals.shape[0])  # for unit variance

    return signals

