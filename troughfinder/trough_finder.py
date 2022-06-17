import numpy as np
from tqdm import tqdm
import healpy as hp


class TroughFinder():
    """Trough Finder
    """

    def __init__(self, density_map, mask_map, verbose=False):

        if density_map.shape[0] != mask_map.shape[0]:
            raise ValueError(
                "density_map and mask_map must have the same size."
            )
        self.density_map = density_map
        self._nside = hp.npix2nside(len(density_map))
        self.mask_map = mask_map

        # Define verbose option
        self._verboseprint = print if verbose else lambda *a, **k: None
        self._verbose = verbose

    def _extand_trough(
        self,
        start_pix,
        current_amp,
        keep_list,
        amp_thresh,
    ):
        """Extand trough

        The goal of this method is to define a region delimited by either a
        saddle point or a mask.\n

        This method take as an input a healpix pixel `start_pix` and will look
        for pixels around that are below the given threshold `amp_thresh`
        (taken as the average density) and also make sure that the amplitude of
        the neighbouring pixel is above `current_amp` (to reach a saddle
        point).\n

        WARNING: Recursive function.\n

        Parameters
        ----------
            start_pix: int
                Starting pixel position on the density map. for the first call
                of this method, it is define as a local minima on the density
                map.
            current_amp: float
                Amplitude of the map at `start_pix` position.
            keep_list: list
                List of pixels that belong to the region. This is updated at
                each call of the function. It has to be initialise with the
                starting pixel as: `keep_list = [start_pix]`.
            amp_thresh: float
                Maximum allowed amplitude. Usually set to the average density
                of the field.
            nside: int
                Healpix `nside` parameter.

        Returns
        -------
            keep_list: list
                List of pixels belonging to the region.
        """

        ra_start, dec_start = hp.pix2ang(self._nside, start_pix, lonlat=True)
        close_pix = hp.get_all_neighbours(
            self._nside,
            ra_start,
            dec_start,
            lonlat=True
        )

        # Out of bound
        close_pix = close_pix[close_pix != -1]

        # Out of mask
        close_pix = close_pix[self.mask_map[close_pix] > 0.99]

        # Below thresh
        close_pix = close_pix[self.density_map[close_pix] < amp_thresh]

        # Stay in current trough
        close_pix = close_pix[self.density_map[close_pix] > current_amp]

        # Already counted
        close_pix = close_pix[
            [check_pix not in keep_list for check_pix in close_pix]
        ]

        if len(close_pix) == 0:
            # If didn't fount new pixels, we stop and return the current list
            return keep_list
        else:
            # If we found new pixels, we restat the process on the new pixels
            for check_pix in close_pix:
                if check_pix in keep_list:
                    continue
                keep_list = np.append(keep_list, check_pix)
                keep_list = self._extand_trough(
                    check_pix,
                    self.density_map[check_pix],
                    keep_list,
                    amp_thresh,
                )
            return keep_list

    def _pix_to_circle(
        self,
        center_pix,
        trough_region,
        start_step=8,
        min_step=1,
    ):
        """Pixels to circle

        Draw a circle which englobe a given pixel region.\n

        We first look for the circle that contain only pixels that belong to
        the given region `trough_region`. Then we grow this circle until the
        majority of pixels we add do not belong to `trough_region` or the step
        is smaller than `min step`.

        Parameters
        ----------
            center_pix: int
                Healpix pixel to use as center for the circle.
            trough_region: list
                List of healpix pixels representing the trough region.
            start_step: int, optional
                Starting step size (in arcmin). [Defaults: 8]
            min_step: int, optional
                Minimum step size (in arcmin). [Defaults: 1]

        Returns
        -------
        radius: float
            Final radius of the circle (in arcmin).
        within_pix: numpy.ndarray
            Array of the pixels within the circle.
        """

        # Check first the maximum radius that contains only pixels belonging
        # to the trough region
        vec = hp.pix2vec(self._nside, center_pix)
        radius = self._arcmin_to_rad(start_step)
        direc = 1
        step = start_step
        while True:
            within_pix = hp.query_disc(self._nside, vec, radius)
            common_pix_frac = len(np.intersect1d(within_pix, trough_region)) \
                / len(within_pix)
            if common_pix_frac >= 1:
                if direc == 1:
                    direc = 1
                    radius += self._arcmin_to_rad(step)
                else:
                    direc = 1
                    step /= 2
                    if step < min_step:
                        break
                    radius += self._arcmin_to_rad(step)
            elif common_pix_frac <= 1:
                if direc == 1:
                    direc = -1
                    step /= 2
                    if step < min_step:
                        break
                    radius -= self._arcmin_to_rad(step)
                else:
                    direc = -1
                    radius -= self._arcmin_to_rad(step)

        # Now we increase the radius step by step until the majoraity of new
        # pixels we add are not from the trough region
        current_pix = hp.query_disc(self._nside, vec, radius)
        step = start_step
        new_radius = radius + self._arcmin_to_rad(step)
        while True:
            within_pix = hp.query_disc(self._nside, vec, new_radius)
            mask = np.ones_like(within_pix, dtype=bool)
            common_pix_frac, ind_current_in_new, _ = np.intersect1d(
                within_pix,
                current_pix,
                return_indices=True,
            )
            mask[ind_current_in_new] = False
            new_pix = within_pix[mask]
            if len(new_pix) == 0:
                radius = new_radius
                new_radius = radius + self._arcmin_to_rad(step)
                continue
            new_pix_frac = len(np.intersect1d(new_pix, trough_region)) \
                / len(new_pix)
            if new_pix_frac >= 0.5:
                radius = new_radius
                new_radius = radius + self._arcmin_to_rad(step)
                current_pix = within_pix
            else:
                step /= 2
                if step < min_step:
                    break
                new_radius = radius + self._arcmin_to_rad(step)

        if self._verbose:
            common_pix_frac = len(np.intersect1d(current_pix, trough_region)) \
                / len(current_pix)
            # print("final frac: ", common_pix_frac)

        radius = np.rad2deg(radius)/60

        return radius, within_pix

    def _get_weighted_prop(self, trough_region):
        """Get weighted properties

        This method found the best center for a given trough region by
        computing the weighted averaged of all the pixels composing this
        region. We use the inverse density as a weight. We then found the best
        circle with this center by calling `_pix_to_center`.

        Parameters
        ----------
            trough_region: numpy.ndarray
                Array of healpix pixels representing the trough region.

        Returns
        -------
            pix_center: int
                Healpix pixel center of the circle.
            vec: tuple
                The coordinates of unit vector defining the circle center (see
                healpy documentation for more details).
            ra_center: float
                RA position of the center (in degree).
            dec_center: float
                DEC position of the center (in degree).
            radius: float
                Radius of the circle (in arcmin).
        """

        ra_region, dec_region = hp.pix2ang(
            self._nside,
            trough_region,
            lonlat=True,
        )
        w_region = 1./self.density_map[trough_region]
        ra_center = np.average(ra_region, weights=w_region)
        dec_center = np.average(dec_region, weights=w_region)

        pix_center = hp.ang2pix(
            self._nside,
            ra_center,
            dec_center,
            lonlat=True,
        )
        vec = hp.pix2vec(self._nside, pix_center)

        radius, _ = self._pix_to_circle(
            pix_center,
            trough_region,
            start_step=0.5,
            min_step=0.001,
        )

        return pix_center, vec, ra_center, dec_center, radius

    def _cleanup_troughs(
        self,
        trough_radius,
        trough_center_pix,
        trough_regions,
    ):
        """Cleanup troughs

        This method go through all defined circles defined by `_pix_to_circle`
        and check if some circle centers are included in others. If that is the
        case, we keep the largest one, we merge the trough regions of all the
        included smaller trough into the large one and re-compute the best
        circle in the enlarge area. To define the center in this region, see
        `_get_weighted_prop`.

        Args:
            trough_radius: numpy.ndarray
                Array of all trough radius (in arcmin).
            trough_center_pix: numpy.ndarray
                Array of all trough centers.
            trough_regions: list
                List of all trough regions defined as list of Healpix pixels.

        Returns:
            _type_: _description_
        """

        # We sort all trough by decreasing radius
        ind_trough_sorted = np.flip(np.argsort(trough_radius))

        removed_trough = np.array([], dtype=np.int64)
        new_troughs = np.array([], dtype=np.int64)
        new_trough_radius = np.array([])
        new_trough_ra = np.array([])
        new_trough_dec = np.array([])
        for ind_trough in tqdm(
            ind_trough_sorted,
            total=len(ind_trough_sorted),
            disable=(not self._verbose)
        ):
            # We check if the trough hasn't been already removed
            if trough_center_pix[ind_trough] in removed_trough:
                continue
            vec = hp.pix2vec(self._nside, trough_center_pix[ind_trough])
            current_pix_center = trough_center_pix[ind_trough]
            current_radius = trough_radius[ind_trough]
            ra_center, dec_center = hp.pix2ang(
                self._nside,
                current_pix_center,
                lonlat=True,
            )
            current_region = np.array([], dtype=int)

            while True:
                within_pix = hp.query_disc(
                    self._nside,
                    vec,
                    self._arcmin_to_rad(current_radius),
                )

                # We check if other trough centers falls in the current one
                # This will always return atleast one center, the one of the
                # trough we are checking
                common_pix = np.intersect1d(within_pix, trough_center_pix)

                # We check if the trough found arn't already removed
                _, already_removed, _d = np.intersect1d(
                    common_pix,
                    removed_trough,
                    return_indices=True
                )

                # We remove from the list the trough that are already removed
                if len(already_removed) != 0:
                    common_pix = np.delete(common_pix, already_removed)

                # At this stage if there is only the current trough center in
                # the circle, we stop and return its properties
                if len(common_pix) == 1:
                    new_troughs = np.append(new_troughs, current_pix_center)
                    new_trough_radius = np.append(
                        new_trough_radius,
                        current_radius
                    )
                    new_trough_ra = np.append(new_trough_ra, ra_center)
                    new_trough_dec = np.append(new_trough_dec, dec_center)
                    break

                # We combine the regions of the overlaping trough
                combined_regions = np.concatenate(
                    [
                        trough_regions[
                            np.where(trough_center_pix == i)[0][0]
                        ] for i in common_pix
                    ]
                )

                # Now we add the region of the trough we are checking
                combined_regions = np.concatenate(
                    (combined_regions, current_region)
                )

                # Remove duplicated pixels
                # That is necessary for second iteration and after
                combined_regions = np.array(list(set(combined_regions)))

                # Remove current trough center from list (if in)
                if len(
                    common_pix[common_pix == trough_center_pix[ind_trough]]
                ) == 1:
                    common_pix = np.delete(
                        common_pix,
                        np.where(common_pix == trough_center_pix[ind_trough])
                    )

                removed_trough = np.concatenate((removed_trough, common_pix))

                # Get new properties for the trough region
                (
                    current_pix_center,
                    vec,
                    ra_center,
                    dec_center,
                    current_radius
                ) = self._get_weighted_prop(combined_regions)

                # New region
                current_region = combined_regions

        return (
            new_troughs,
            new_trough_ra,
            new_trough_dec,
            new_trough_radius,
            removed_trough,
        )

    @staticmethod
    def _arcmin_to_rad(arcmin):
        """Arcmin to radian

        Convert arcminutes to radians

        Args:
            arcmin: float
                Angle in arcmin

        Returns:
            float
            Angle in radian
        """
        return np.deg2rad(arcmin/60)

    def run(
        self,
        density_threshold=1,
        do_cleanup=True,
        keep_before_cleanup=False,
    ):
        """Run
        """

        if (not do_cleanup) & (not keep_before_cleanup):
            raise ValueError(
                "Either 'do_cleanup' or 'keep_before_cleanup' must be true,"
                " otherwise the method has no outputs."
            )

        # Get mean density
        mean_dens = np.average(self.density_map, weights=self.mask_map)

        # Get location of local minima and their amplitude
        _, min_pix_dens, _ = hp.hotspots(self.density_map)
        amp_min_pix = self.density_map[min_pix_dens]
        mask_good_minima = (amp_min_pix < mean_dens/density_threshold) \
            & (self.mask_map[min_pix_dens] >= 1)

        trough_pix_before = min_pix_dens[mask_good_minima]
        trough_region_pix = []
        if keep_before_cleanup:
            trough_ra_before = np.array([])
            trough_dec_before = np.array([])
            trough_radius_before = np.array([])
        for tmp_minima_pix in tqdm(
            trough_pix_before,
            total=len(trough_pix_before),
            disable=(not self._verbose),
        ):
            # Found the trough region defined by saddle point or mask
            raw_trough_pix = self._extand_trough(
                tmp_minima_pix,
                self.density_map[tmp_minima_pix],
                [tmp_minima_pix],
                mean_dens/density_threshold,
            )
            trough_region_pix.append(raw_trough_pix)

            # First estimate of the trough radius
            trough_rad, _ = self._pix_to_circle(
                tmp_minima_pix,
                raw_trough_pix,
                start_step=0.5,
                min_step=0.001
            )

            ra_tmp, dec_tmp = hp.pix2ang(
                self._nside,
                tmp_minima_pix,
                lonlat=True,
            )

            if keep_before_cleanup:
                trough_ra_before = np.append(trough_ra_before, ra_tmp)
                trough_dec_before = np.append(trough_dec_before, dec_tmp)
                trough_radius_before = np.append(
                    trough_radius_before,
                    trough_rad
                )

        if do_cleanup:
            (
                trough_pix,
                trough_ra,
                trough_dec,
                trough_radius,
                removed_trough
            ) = self._cleanup_troughs(
                trough_radius_before,
                trough_pix_before,
                trough_region_pix,
            )

        if do_cleanup & keep_before_cleanup:
            return (
                trough_pix,
                trough_ra,
                trough_dec,
                trough_radius,
                removed_trough,
                trough_pix_before,
                trough_ra_before,
                trough_dec_before,
                trough_radius_before
            )
        elif do_cleanup & (not keep_before_cleanup):
            return (
                trough_pix,
                trough_ra,
                trough_dec,
                trough_radius,
                removed_trough,
            )
        else:
            return (
                trough_pix_before,
                trough_ra_before,
                trough_dec_before,
                trough_radius_before
            )
