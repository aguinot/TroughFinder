import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as ast_u

import healpy as hp

from scipy.spatial import cKDTree, SphericalVoronoi

from .healpy_utils import bin_map


class VoronoiDensityField():
    """
    """

    def __init__(self, ra, dec, pair_thresh=1e-10, verbose=False):

        self._ra = ra
        self._dec = dec
        self._pair_thresh = pair_thresh

        # Define verbose option
        self._verboseprint = print if verbose else lambda *a, **k: None

    def _convert_to_catesian(self):
        """ Convert to cartesian

        Convert Ra, Dec positions to cartesian on the unit sphere.

        """

        sphe_coord = SkyCoord(
            ra=self._ra*ast_u.degree,
            dec=self._dec*ast_u.degree
        )
        cart_coord = sphe_coord.cartesian
        self._cart_point = np.array([
            cart_coord.x.value,
            cart_coord.y.value,
            cart_coord.z.value,
        ]).T

    def _pre_checks(self):
        """Pre checks

        Removes pairs from the dataset to avoid
        `scipy.spatial.SphericalVoronoi` to raise anerror.
        From the pairs `[Obj1, Obj2]` we always remove the second element from
        the list `Obj2`.

        """

        tree = cKDTree(self._cart_point)
        pairs = tree.query_pairs(self._pair_thresh)
        pairs = np.array(list(pairs))
        mask_duplicate = np.ones_like(self._ra, dtype=bool)
        if len(pairs) != 0:
            mask_duplicate[pairs[:, 1]] = False

        self._cart_point = self._cart_point[mask_duplicate]
        self._ra = self._ra[mask_duplicate]
        self._dec = self._dec[mask_duplicate]

        self._verboseprint(
            "Removed objects during pre-check: ", pairs.shape[0]
        )

    def _compute_voronoi(self):
        """Compute Voronoi

        Make the Voronoi tesselation on the unit sphere and get the regions'
        area in arcmin^2.

        """

        self.sv = SphericalVoronoi(
            self._cart_point,
            radius=1,
            center=np.array([0, 0, 0]),
            threshold=self._pair_thresh,
        )

        voro_area = self.sv.calculate_areas()
        self.voronoi_area = voro_area * (60 * np.rad2deg(1))**2

    def get_healpix_map(self, nside, ignore_low_density=False):
        """Get healpix map

        Interpolate voronoi tesselation on healpix pixels. We make this
        assumption that all the voronoi cells are smaller than each healpix
        pixels so we can just average the voronoi densities in those pixels
        without further consideration.

        WARNING : If the field does not have an high enough density the
        approximation made here does not old any more and we could get weird
        results! We raise an error if that is the case and
        `ignore_low_density==False`.

        Parameters
        ----------
        nside: int
            Healpix `nside` parameter.
        ignore_low_density: bool
            If `True` will ignore the case where the voronoi cells are larger
            than the healpix pixels (less than 2 voronoi cells per healpix
            pixels). [Default: False]

        Returns
        -------
        voro_dens: numpy.ndarray
            Healpix map representing Voronoi density field.
        hp_dens: numpy.ndarray
            Healpix number count.
        """

        # Healpix pixel area in arcmin
        hp_pixarea = hp.nside2pixarea(nside, degrees=True)*3600
        # We take the median to avoid large area due to the edge effect
        # TODO: deal with the edge effect. Not a bid issue with a good masking.
        median_voro_area = np.median(self.voronoi_area)
        n_voro_per_hp_pix = int(hp_pixarea/median_voro_area)

        self._verboseprint(
            "Average number of voronoi cells per healpix pixels:"
            " {}".format(n_voro_per_hp_pix)
        )
        if (n_voro_per_hp_pix <= 2) & ignore_low_density:
            raise ValueError(
                "Voronoi cells are to large for the healpix pixels."
                " Found {} cells per pixels".format(n_voro_per_hp_pix)
            )

        hp_dens, voro_dens = bin_map(
            self._ra,
            self._dec,
            nside,
            weights=1./self.voronoi_area,
            method='average',
        )

        return voro_dens, hp_dens
