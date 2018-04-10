"""
Utilities to compute a brain mask from EPI images
"""

import numpy as np
from scipy import ndimage
import nibabel

###############################################################################
# Time series extraction
###############################################################################

def apply_mask(niimgs, mask_img, dtype=np.float32,
                     ensure_finite=True):
    """Extract signals from images using specified mask.

    Read the time series from the given nifti images or filepaths,
    using the mask.

    Parameters
    -----------
    niimgs: list of 4D nifti images
        Images to be masked. list of lists of 3D images are also accepted.

    mask_img: niimg
        3D mask array: True where a voxel should be used.

    ensure_finite: bool
        If ensure_finite is True (default), the non-finite values (NaNs and
        infs) found in the images will be replaced by zeros.

    Returns
    --------
    session_series: numpy.ndarray
        2D array of series with shape (image number, voxel number)

    """

    if isinstance(mask_img, str):
        mask_img = nibabel.load(mask_img)
    mask_data = mask_img.get_data().astype(bool)
    mask_affine = mask_img.get_affine()


    if isinstance(niimgs, str):
        niimgs = nibabel.load(niimgs)
    affine = niimgs.get_affine()[:3, :3]

    if not np.allclose(mask_affine, niimgs.get_affine()):
        raise ValueError('Mask affine: \n%s\n is different from img affine:'
                         '\n%s' % (str(mask_affine),
                                   str(niimgs.get_affine())))

    if not mask_data.shape == niimgs.shape[:3]:
        raise ValueError('Mask shape: %s is different from img shape:%s'
                         % (str(mask_data.shape), str(niimgs.shape[:3])))

    # All the following has been optimized for C order.
    # Time that may be lost in conversion here is regained multiple times
    # afterward
    data = niimgs.get_data()
    series = np.asarray(data)
    del data, niimgs  # frees a lot of memory

    return series[mask_data].T


def unmask(X, mask_img, order="C"):
    """Take masked data and bring them back to 3D (space only).

    Parameters
    ==========
    X: numpy.ndarray
        Masked data. shape: (samples,)

    mask_img: niimg
        3D mask array: True where a voxel should be used.
    """

    if isinstance(mask_img, str):
        mask_img = nibabel.load(mask_img)
    mask_data = mask_img.get_data().astype(bool)

    data = np.zeros(mask_data.shape + (X.shape[0],), dtype=X.dtype, order=order)
    data[mask_data, :] = X.T
    return data

