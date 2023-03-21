# -*- coding: utf-8 -*-

'''
Created on 30 aout 2016, PH
Modified on 8 Oct. 2018, LVG multiple parameters matching
Modified on 2 Mar. 2023, LVG cleaning and documentation, merge with the work of Victor Nussbaum

@author: Hagenmuller P.
@author: Viallon Galinier L.

Set of functions that relate prepare the data and call DTW_CCore routines.

In this version, data have to have the same size. Nan are taken as data and not as end of dataset.
'''
import os.path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

_here = os.path.dirname(os.path.realpath(__file__))

try:
    from .DTW_CCore import DTW_multi
except ModuleNotFoundError:
    _filepath = os.path.join(_here, 'DTW_CCore.so')
    print("Impossible import of DTW_CCore ({})\n"
          "Make sure that you have compiled the DTW C program. See documentation.".format(_filepath))
    raise

U2undef = 65000


def fromIDtoGrid_multi(sigma_s, depth_grid, IDs):
    """
    Function to provide shifted snow properties and depths from the optimal path computed
    by DTW and stored in IDs matrix.

    Returns both the shifted snow properties and the shifted depth

    :param sigma_s: Snow parameters to shift
    :type sigma_s: numpy array (2, N, P)
    :param depth_grid: Depth grid
    :type depth_grid: numpy array (N)

    :retuns: Moved sigma_s and depth moved
    :rtype: numpy arrays (N, P) and (2, N)
    """
    dim_sigma = sigma_s.shape[1]
    depth_moved = np.zeros((2, sigma_s.shape[2]))
    sigma_moved = np.zeros_like(sigma_s)

    # only for two profiles
    ok = (IDs[:, 0] < U2undef) * (IDs[:, 1] < U2undef)
    depth_moved[0] = depth_grid
    depth_moved[1] = np.interp(depth_grid, depth_grid[IDs[ok, 1]][::-1],
                               depth_grid[IDs[ok, 0][::-1]])
    sigma_moved[0] = sigma_s[0]
    for i in range(dim_sigma):
        sigma_moved[1, i, :] = np.interp(depth_grid, depth_moved[1], sigma_s[1, i, :])

    return sigma_moved, depth_moved


def fit_to_ref_multi(sigma, sigma_ref, coeffs, depth_grid, partial, coutdecal=0, unbias=False):
    """
    Main function to match two snow profiles. The profiles are described by the depth grid,
    and th eproperties of the two profiles: the reference one (sigma_ref) and the profile to
    be matched (sigma).

    Prerequisites:

     - Profiles should share the same depth grid (thus have the same snow depth)
       and be projected on a grid that allow reattributions of points (hence sufficiently
       close, a typical value is around the mm for typical snow cover simulations).

    Let's note N the number of elements on the vertical grid and P the number of features to match.

    :param sigma: Profile to be matched
    :type sigma: numpy array (P, N)
    :param sigma_ref: Reference profile
    :type sigma: numpy array (P, N)
    :param coeffs: Coefficients
    :type coeffs: numpy array (P, 3)
    """
    if isinstance(unbias, bool):
        unbias_bool = unbias
        unbias_columns = [0, 1, 2, 3]
    elif isinstance(unbias, list):
        unbias_bool = True
        unbias_columns = False
    else:
        unbias_bool = False

    # Shapes
    N = sigma.shape[1]
    P = sigma.shape[0]

    # Create objects further used by DTW
    IDs = np.zeros(shape=(2 * N - 1, 2), dtype='u2')
    D = np.empty((N, N), dtype='float32')
    B = np.empty((N, N), dtype='i1')
    sigma2 = sigma.copy()

    if unbias_bool:
        for i in unbias_columns:
            msr = np.nanmean(sigma_ref[i, :])
            ms1 = np.nanmean(sigma[i, :])
            sigma2[i, :] = sigma2[i, :] - (ms1 - msr)

    # Call DTW
    DTW_multi(sigma_ref.astype('f4'), sigma2.astype('f4'), coeffs.astype('f4'), D, B, IDs, partial, coutdecal)

    sigma_s = np.array([sigma_ref, sigma])
    sigma_moved, depth_moved = fromIDtoGrid_multi(sigma_s, depth_grid, IDs)

    return sigma_moved, depth_moved


def downsample_profile(depth, value, depth_grid,
                       depth_smooth=0, ratio_smoothing=1,
                       is_sorted=False,
                       left_value=None, right_value=None):
    """
    Re-samples a snow profile with a high resolution (higher than the target one).
    The profile is the value of one parameter as a function of depth. It is resampled
    onto another depth array `depth_grid`.

    Note that this function is used for downsampling, with an optional smoothing of the initial profile.

    :param value: the variable of the profile (e.g. density, hardness)
    :param depth: the depth of the profile corresponding to value
    :param depth_grid: the depth on which the profile (value, depth) is resampled (units have to be coherent with depth)
    :param depth_smooth: (optional) size (same unit as param depth) of the Gaussian kernel
    used to smoothen the initial profile. Generally set to the typical resolution of depth_grid.
    :param ratio_smoothing: used to re-interpolate the initial profile on higher sampling rate
    in case of irregular depth spacing.
    :param is_sorted: set to False if depth is not sorted increasingly
    :return: the variable value resampled on depth_grid

    :type value: numpy array (1D)
    :type depth: numpy array (1D)
    :type depth_grid: numpy array (1D)
    :type depth_smooth: float
    :type ratio_smoothing: int
    :type is_sorted: boolean
    :rtype: numpy array of the same dtype as value

    :warning:: if the depth array is not sorted increasingly, set is_sorted to False,
    elsewise it will give wrong results.

    :Example:

    >>> depth = np.linspace(0,1,100)
    >>> value = np.sin(depth)
    >>> depth_grid = np.linspace(0,0.8,20)
    >>> value_on_grid = downsample_profile(value,depth,depth_grid,0.8/20.,1)
    >>> print np.abs(value_on_grid - np.sin(depth_grid)).max() < 0.02
    True
    """

    # Sorting the initial array
    if not is_sorted:
        sorted_index = np.argsort(depth)
        depth = depth[sorted_index]
        value = value[sorted_index]

    # In case of down-sampling
    if(depth_smooth > 0):
        depth_min = np.nanmin(depth)
        depth_max = np.nanmax(depth)
        num = int(ratio_smoothing * (depth_max - depth_min) / (1.0 * depth_smooth))

        depth_grid_smooth = np.linspace(depth_min, depth_max, num + 1)
        depth_grid_smooth_step = depth_grid[1] - depth_grid[0]
        value_grid_smooth = np.interp(depth_grid_smooth, depth, value)
        value_smooth = gaussian_filter(value_grid_smooth, sigma=1.0 * depth_smooth / depth_grid_smooth_step)

    else:
        depth_grid_smooth = depth
        value_smooth = value

    # Default values initialisation
    if not left_value:
        left_value = value[0]
    if not right_value:
        right_value = value[-1]

    return np.interp(depth_grid, depth_grid_smooth, value_smooth, right=right_value, left=left_value)


def oversample_profile(depth, value, depth_grid,
                       is_sorted=False, kind='next'):
    """
    Re-samples a snow profile with a low resolution (lower than the target one).
    The profile is the value of one parameter as a function of depth. It is resampled
    onto another depth array `depth_grid`.

    Note that this function is used for oversampling. for downsampling, you need
    an additional smoothing of the intial profile. Please refer to function
    `downsample_profile`.

    :param value: the variable of the profile (e.g. density, hardness)
    :param depth: the depth of the profile corresponding to value
    :param depth_grid: the depth on which the profile (value, depth) is resampled
    :param is_sorted: set to False if depth is not sorted increasingly
    :param kind: Kind of oversampling. Please refer to scipy.interpolate.interp1d function
    documentation. Basically, `next` will be suitable if your depth are top depth,
    `previous` if you provide bottom depth or `nearest` if you provide middle depth
    (in case of regular grid only for `depth`).
    :return: the variable value resampled on depth_grid

    :type value: numpy array (1D)
    :type depth: numpy array (1D)
    :type depth_grid: numpy array (1D)
    :type is_sorted: boolean
    :rtype: numpy array of the same dtype as value

    :warning:: if the depth array is not sorted increasingly, set is_sorted to False,
    elsewise it will give wrong results.

    :Example:

    >>> depth = np.linspace(0,1,100)
    >>> value = np.sin(depth)
    >>> depth_grid = np.linspace(0,0.8,20)
    >>> value_on_grid = oversample_profile(value,depth,depth_grid,0.8/20.,1)
    >>> print np.abs(value_on_grid - np.sin(depth_grid)).max() < 0.02
    True
    """

    # Sorting the initial array
    if not is_sorted:
        sorted_index = np.argsort(depth)
        depth = depth[sorted_index]
        value = value[sorted_index]

    f = interp1d(depth, value, kind=kind, fill_value="extrapolate")
    return f(depth_grid)


def scaling_profile(value, mean: float = None, sigma: float = None):
    """
    Scale the profile to set mean at zero and variance to 1 by substracting the
    mean value and dividing by the variance (sigma).

    :param value: The numpy array of the value to modify
    :type value: numpy array ((1,) N)
    :param mean: if None, compute the mean of value
    :type mean: float or None
    :param sigma: if None, compute the variance of value
    :type sigma: float or None
    """
    if mean is None:
        mean = np.nanmean(value)
    if sigma is None:
        sigma = np.var(value)
    return (value - mean) / sigma
