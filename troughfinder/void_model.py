import numpy as np

from scipy.misc import derivative

import pyccl as ccl


class CosmicVoidModel():
    """Cosmic Void Model

    Class to compute the cosmic void size function.\n
    This code is a python transcription of the CosmoBolognaLib in c++. We make
    use of the CCL librairie to handle the cosmology part. The orginal code is
    accessible at: https://github.com/federicomarulli/CosmoBolognaLib \n
    Paper: https://arxiv.org/abs/1703.07848

    Parameters
    ----------
        Omega_m: float
            Total matter density fraction.
        Omega_c: float
            Cold Dark Matter density fraction.
        Omega_b: float
            Baryonic matter density fraction.
        h: float
            Reduced Hubble constant.
        n_s: float
            Primordial scalar perrturbation spectral index.
        sigma8: float
            Variance of the matter density perturbation at 8 Mpc/h scale.
        A_s: float
            Power spectrum normalization.
        kwargs: dict
            Any other arguments that can be pass to `pyccl.Cosmology`.
    """

    def __init__(
        self,
        Omega_m=None,
        Omega_c=None,
        Omega_b=None,
        h=None,
        n_s=None,
        sigma8=None,
        A_s=None,
        **kwargs,
    ):

        if not isinstance(Omega_m, type(None)) and \
                not isinstance(Omega_c, type(None)):
            raise ValueError(
                "Exactly one of Omega_m or Omega_c should be given, not both."
                " We assume Omega_m = Omega_c + Omega_b"
            )
        if isinstance(Omega_b, type(None)):
            # Raise ccl error
            raise ValueError(
                "Necessary parameter 'Omega_b' was not set (or set to None)."
            )
        if not isinstance(Omega_m, type(None)):
            Omega_c = Omega_m - Omega_b

        # Init cosmology
        # We let ccl handles te errors
        self.cosmo = ccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            n_s=n_s,
            sigma8=sigma8,
            A_s=A_s,
            **kwargs,
        )

    def size_function(
        self,
        RV,
        redshift,
        b_eff,
        model="Vdn",
        slope=0.854,
        offset=0.420,
        deltav_NL=-0.795,
        delta_c=1.686,
    ):

        """Size function

        the void size function

        Parameter
        ---------
            RV: float or numpy.array
                Radius (in Mpc).
            redshift: float
                Redshift.
            b_eff: float
                Effective tracer bias.
            model: str
                Size function model., must be in ['Vdn', 'SvdW', 'lnear'].
                [Default: 'Vdn']
            slope: float
                    First coefficient to convert the effective bias.
                    [Dfault: 0.854]
            offset: float
                Second coefficient to convert the effective bias.
                [Default: 0.420]
            deltav_NL: float
                Non-Linear negative density threshold.
            delta_c: float, optional
                Positive density threshold. [Defalut: 1.686]
        Returns
        -------
            float or numpy.ndarray:
                The void size function.
        """

        del_v = self.deltav_L(deltav_NL, b_eff, slope, offset)
        if (model == 'Vdn') | (model == 'SvdW'):
            RL = RV/self.r_rL(del_v)
        elif model == 'linear':
            RL = RV
        else:
            raise ValueError("model must be in ['Vdn', 'SvdW', 'linear']")

        fact = self.cosmo.growth_factor(1/(1+redshift))

        sigmaR = self.cosmo.sigmaR(RL, 1.)
        sigmaRz = sigmaR*fact
        SSSR = sigmaRz**2

        dnsigma2R = self._d1sigma2R(RL)
        f_nu = self.f_nu(sigmaRz, del_v, delta_c)

        Dln_SigmaR = dnsigma2R * (RL/(2.*SSSR))*fact**2
        if model == 'Vdn':
            return f_nu/self._volume_sphere(RV)*np.abs(Dln_SigmaR)
        else:
            return f_nu/self._volume_sphere(RL)*np.abs(Dln_SigmaR)

    def f_nu(self, SS, del_v=-0.795, del_c=1.686):
        r"""f_nu

        Fraction of the Univers occupied by cosmic voids (approximation).\n

        .. math::
            f_{\mathrm{ln}~\sigma}(\sigma)

        NOTE:
            Implementation from CosmoBolognaLib.

        Parameters
        ----------
            SS: float or numpy.ndarrray
                Variance of the linear density field.
            del_v: float, optional
                Non-linear Negative density threshold (shell-crossing
                threshold).
                [Default: -0.795]
            del_c: float, optional
                Positive density threshold. [Defalut: 1.686]

        Returns
        -------
            f_nu: float or numpy.ndarray
                Fraction of the Univers occupied by cosmic voids.
        """

        radnu = np.abs(del_v)/SS
        nu = radnu**2.
        DDD = np.abs(del_v)/(del_c+np.abs(del_v))
        xx = DDD/radnu

        # x <= 0.278
        ff = np.zeros_like(xx, dtype=np.float64)
        mm = xx <= 0.276
        ff[mm] = np.sqrt(2./np.pi)*radnu[mm]*np.exp(-0.5*nu[mm])

        # x > 0.278
        inv_mm = np.invert(mm)
        j = (
            np.array([np.arange(1, 5, dtype=np.float64)])
            * np.ones((len(xx[inv_mm]), 1))
        ).T
        ff[inv_mm] = 2 * np.sum(
            np.exp(-(j*np.pi*xx[inv_mm])**2. / 2.)
            * j * np.pi * xx[inv_mm]**2 * np.sin(j*np.pi*DDD),
            axis=0
        )

        return ff

    def _d1sigma2R(self, R):
        """

        First derivative of sigma_R^2.

        Parameters
        ----------
            R: float or numpy.ndarray
                Radius (in Mpc).

        Returns
        -------
            float or numpy.ndarray
                First derivative of sigma_R^2 (in Mpc).
        """

        return derivative(self._sigma2R, R, 1e-6, order=9)

    def _sigma2R(self, R, z=0.):
        """

        Variance in a top-hat sphere of radius R in Mpc.

        Parameters
        ----------
            R: float or numpy.ndarray
                Radius (in Mpc).
            z: float, optional
                Redshift. [Defaults: 0]

        Returns
        -------
            float or numpy.ndarray
                Variance in the density field in top-hat sphere (in Mpc^2).
        """

        return self.cosmo.sigmaR(R, self._z_to_a(z))**2

    def deltav_L(self, deltav_NL, b_eff, slope=0.854, offset=0.420):
        """deltav linear

        Linear negative denisty threshold.

        Parrameters
        -----------
            deltav_NL: float
                Non-Linear negative density threshold.
            b_eff: float
                Effective tracer bias.
            slope: float
                First coefficient to convert the effective bias.
                [Dfault: 0.854]
            offset: float
                Second coefficient to convert the effective bias.
                [Default: 0.420]

        Returns
        -------
            float:
                Lienar negative density threshold.
        """

        if b_eff == 1:
            slope = 1
            offset = 1
        return 1.594*(1.-(1+deltav_NL/(slope*b_eff+offset))**(-1./1.594))

    def r_rL(self, deltav_L):
        """

        Expansion factor.

        Parameters
        ----------
            deltav_L: float
                Linea negative dansity threshold.

        Returns
        -------
            float:
                Expansion factor
        """

        return ((1.-deltav_L/1.594)**(-1.594))**(-1./3.)

    @staticmethod
    def _z_to_a(z):
        """z to a

        Convert redshift `z` to scale factor `a`.

        Parameters
        ----------
            z: float or numpy.ndarray
                Redshift.

        Returns
        -------
            a: float or numpy.ndarrray
                Scale factor.
        """

        return 1./(1. + z)

    @staticmethod
    def _volume_sphere(R):
        """

        Compute the volume of a sphere.

        Args:
            R: float or numpy.array
                Radius.

        Returns:
            float or numpy.ndarray
                Volume of the sphere.
        """
        return 4./3.*np.pi*R**3
