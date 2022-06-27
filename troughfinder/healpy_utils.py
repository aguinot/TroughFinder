import healpy as hp
import numpy as np


def eq2ang(ra, dec):
    """
    convert equatorial ra,dec in degrees to angular theta, phi in radians.

    Parameters
    ----------
    ra: scalar or array
        Right ascension in degrees.
    dec: scalar or array
        Declination in degrees.

    Returns
    -------
    theta,phi: tuple
        theta = pi/2-dec*D2R # in [0,pi]
        phi   = ra*D2R       # in [0,2*pi]

    """

    dec = dec*np.pi/180
    ra = ra*np.pi/180
    theta = np.pi/2 - dec
    phi = ra

    return theta, phi


def bin_map(ra, dec, nside, weights=None, method='average'):
    """Bin map
    Computes the shear map by binning the catalog according to pixel_index.
    Either nx,ny or npix must be provided.

    Parameters
    ----------
    ra: numpy.ndarray
        Right Ascension of galaxies (in degree).
    dec: numpy.ndarray
        Declination of galaxies (in degree).
    nside: int
        Healpix `nside` parameter.
    weights: numpy.ndarray, optional
        Weight to apply to the pixelisation.
        (The raw number count is returned if `type(method) != numpy.ndarray`).
    method: str, numpy.ndarray
        Method to use when `weights != None`. Can also provide a
        `numpy.ndarray` to use for the normalization.\n
         - If `'average`': the sum of the weights in each pixels will be
         normalized by the number count. [Default]
         - If `'sum'`: We return the sum of the weights in each pixels.
         - If `numpy.ndarray`: will use this array for the normalization.

    Returns
    -------
    wmap: numpy.ndarray
        Weighted map (if `weights != None`).
    nmap: numpy.ndarray
        Number of galaxies per pixels.
    """

    theta, phi = eq2ang(ra, dec)
    pixel_index = hp.ang2pix(nside, theta, phi)

    # Bin the shear catalog
    npix = hp.nside2npix(nside)

    if not isinstance(weights, type(None)):
        Wmap = np.bincount(
            pixel_index,
            weights=weights,
            minlength=npix
        )
        if isinstance(method, str):
            Nmap = np.bincount(pixel_index, minlength=npix)
            if method.lower() == 'average':
                mask_ind = Nmap > 0
                Wmap[mask_ind] /= Nmap[mask_ind]
            elif method.lower() == 'sum':
                pass
            else:
                raise ValueError(
                    "method must be in ['average', 'sum'],"
                    " got {}".format(method)
                )
            return Nmap, Wmap
        elif isinstance(method, np.ndarray):
            if len(method.shape) != 1:
                raise ValueError(
                    "If the provided is normalization is a numpy.ndarray, it"
                    " must be of dim 1, found {}".format(len(method.shape))
                )
            if method.shape[0] != Wmap.shape[0]:
                raise ValueError(
                    "If the provided is normalization is a numpy.ndarray, it"
                    " must be of same shape as the created map:"
                    " {}, got {}".format(Wmap.shape[0], method.shape[0])
                )
            mask_ind = method > 0
            Wmap[mask_ind] /= method[mask_ind]

            return Wmap


def smooth_map(hp_map, mask_map, sigma, thresh_mask=1., kind='LogGauss'):
    """Smooth map

    Apply a smoothing on healpix map. At the moment can apply a simple
    `gaussian` filtering or `LogGaussian`.\n
    The `LogGaussian` filtering consist at computing the `log10` of the map
    before applying the Gaussian kernel. Then we come back to normal space.

    Parameters
    ----------
        hp_map: numpy.ndarray
            Healpix map.
        mask_map: numpy.ndarray
            Healpix map of the mask.
        sigma: float
            Sigma of the Gaussian kernel (in arcmin).
        thresh_mask: float
            Every pixel in hp_map which have a mask value below the threshold
            will be masked. [Defaults: 1.]
        kind: str, optional
            Kind of smoothing to apply in ['Gauss', 'LogGauss'].
            [Defaults: 'LogGauss']

    Returns
    -------
    smooth_map: numpy.ndarray
        Smoothed map.
    """

    if kind.lower() == 'gauss':
        smooth_map = hp.smoothing(hp_map, sigma=np.deg2rad(sigma/60))
    elif kind.lower() == 'loggauss':
        log_map = np.zeros_like(hp_map)
        log_map[hp_map != 0] = np.log10(hp_map[hp_map != 0])
        gauss_log_map = hp.smoothing(log_map, sigma=np.deg2rad(sigma/60))
        smooth_map = 10**(gauss_log_map)
        # We set back masked pixels to 0
        # Those might be different than input due to smoothing
        smooth_map[mask_map <= thresh_mask] = 0.
    else:
        raise ValueError(
            "kind must be in ['Gauss', 'LogGauss'], got {}".format(kind)
        )

    return smooth_map
