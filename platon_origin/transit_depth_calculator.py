from pkg_resources import resource_filename
import matplotlib.pyplot as plt
import scipy
import numpy as np

from . import _cupy_numpy as xp
from . import _hydrostatic_solver
from .abundance_getter import AbundanceGetter
from ._species_data_reader import read_species_data
from . import _interpolator_3D
from ._tau_calculator import get_line_of_sight_tau
from ._binning_optimized import bin_ktables_vectorized
from .constants import k_B, AMU, M_sun, Teff_sun, G, h, c
from ._get_data import get_data
from ._mie_cache import MieCache
from .errors import AtmosphereError
from ._atmosphere_solver import AtmosphereSolver
from .params import NUM_LAYERS


class TransitDepthCalculator:
    def __init__(self, include_condensation=True, ref_pressure=1e5, method='xsec', include_opacities=["CH4", "CO2", "CO", "H2O", "H2S", "HCN", "K", "Na", "NH3", "SO2", "TiO", "VO"], downsample=1):
        '''
        All physical parameters are in SI.

        Parameters
        ----------
        include_condensation : bool
            Whether to use equilibrium abundances that take condensation into
            account.
        ref_pressure : float
            The planetary radius is defined as the radius at this pressure
        method : string
            "xsec" for opacity sampling, "ktables" for correlated k
        '''
        self.atm = AtmosphereSolver(include_condensation, ref_pressure, method, include_opacities, downsample)
        self._xsec_bin_bounds = None
        self._xsec_bin_left_indices = None
        self._xsec_bin_right_indices = None

    def change_wavelength_bins(self, bins):
        """Specify wavelength bins, instead of using the full wavelength grid
        in self.lambda_grid.  This makes the code much faster, as
        `compute_depths` will only compute depths at wavelengths that fall
        within a bin.

        Parameters
        ----------
        bins : array_like, shape (N,2)
            Wavelength bins, where bins[i][0] is the start wavelength and
            bins[i][1] is the end wavelength for bin i. If bins is None, resets
            the calculator to its unbinned state.

        Raises
        ------
        NotImplementedError
            Raised when `change_wavelength_bins` is called more than once,
            which is not supported.
        """
        self.atm.change_wavelength_bins(bins)
        self._xsec_bin_bounds = None
        self._xsec_bin_left_indices = None
        self._xsec_bin_right_indices = None
        if self.atm.wavelength_bins is not None and self.atm.method == "xsec":
            self._xsec_bin_bounds = []
            for start, end in self.atm.wavelength_bins:
                left = int(xp.cpu(xp.searchsorted(self.atm.lambda_grid, start)))
                right = int(xp.cpu(xp.searchsorted(self.atm.lambda_grid, end)))
                self._xsec_bin_bounds.append((left, right))
            left_indices = [pair[0] for pair in self._xsec_bin_bounds]
            right_indices = [pair[1] for pair in self._xsec_bin_bounds]
            self._xsec_bin_left_indices = xp.asarray(left_indices, dtype=int)
            self._xsec_bin_right_indices = xp.asarray(right_indices, dtype=int)
        

    def _get_binned_corrected_depths(self, depths, T_star, T_spot,
                                     spot_cov_frac, blackbody=False, n_gauss=10):
        depths = xp.asarray(depths)
        unbinned_lambdas = self.atm.lambda_grid
        stellar_spectrum, correction_factors = self.atm.get_stellar_spectrum(
            T_star, T_spot, spot_cov_frac, blackbody)
        
        #Step 1: do a first binning if using k-coeffs; first binning is a
        #no-op otherwise
        if self.atm.method == "ktables":
            _, weights = scipy.special.roots_legendre(n_gauss)
            weights = xp.asarray(weights / 2)
            intermediate_depths = bin_ktables_vectorized(depths, weights, n_gauss)
            intermediate_lambdas = unbinned_lambdas[::n_gauss]
            intermediate_stellar_spectrum = stellar_spectrum[::n_gauss]
            intermediate_correction_factors = correction_factors[::n_gauss]
            
        elif self.atm.method == "xsec":
            intermediate_lambdas = unbinned_lambdas
            intermediate_depths = depths
            intermediate_stellar_spectrum = stellar_spectrum
            intermediate_correction_factors = correction_factors
        else:
            assert(False)                  
                
        if self.atm.wavelength_bins is None:
            return xp.array(intermediate_lambdas),\
                xp.array(intermediate_depths * intermediate_correction_factors),\
                xp.array(intermediate_stellar_spectrum),\
                xp.array(intermediate_lambdas),\
                xp.array(intermediate_depths * intermediate_correction_factors),\
                xp.array(intermediate_stellar_spectrum),\
                xp.array(intermediate_correction_factors)
                        
        binned_wavelengths = []
        binned_depths = []
        binned_stellar_spectrum = []

        if self.atm.method == "xsec" and self._xsec_bin_bounds is not None:
            bin_ranges = self._xsec_bin_bounds
        else:
            bin_ranges = []
            for start, end in self.atm.wavelength_bins:
                l = int(xp.cpu(xp.searchsorted(intermediate_lambdas, start)))
                r = int(xp.cpu(xp.searchsorted(intermediate_lambdas, end)))
                bin_ranges.append((l, r))

        for l, r in bin_ranges:
            
            binned_wavelengths.append(xp.mean(intermediate_lambdas[l:r]))
            binned_depth = xp.average(intermediate_depths[l:r] * intermediate_correction_factors[l:r],
                                      weights=intermediate_stellar_spectrum[l:r])
            binned_depths.append(binned_depth)
            binned_stellar_spectrum.append(xp.median(intermediate_stellar_spectrum[l:r]))

        return xp.asarray(binned_wavelengths), xp.asarray(binned_depths), xp.asarray(binned_stellar_spectrum), xp.asarray(intermediate_lambdas), xp.asarray(intermediate_depths), xp.asarray(intermediate_stellar_spectrum), xp.asarray(intermediate_correction_factors)

    def _get_binned_depths_only(self, depths, T_star, T_spot,
                                spot_cov_frac, blackbody=False, n_gauss=10):
        depths = xp.asarray(depths)
        unbinned_lambdas = self.atm.lambda_grid
        stellar_spectrum, correction_factors = self.atm.get_stellar_spectrum(
            T_star, T_spot, spot_cov_frac, blackbody)

        if self.atm.method == "ktables":
            _, weights = scipy.special.roots_legendre(n_gauss)
            weights = xp.asarray(weights / 2)
            intermediate_depths = bin_ktables_vectorized(depths, weights, n_gauss)
            intermediate_lambdas = unbinned_lambdas[::n_gauss]
            intermediate_stellar_spectrum = stellar_spectrum[::n_gauss]
            intermediate_correction_factors = correction_factors[::n_gauss]
        elif self.atm.method == "xsec":
            intermediate_lambdas = unbinned_lambdas
            intermediate_depths = depths
            intermediate_stellar_spectrum = stellar_spectrum
            intermediate_correction_factors = correction_factors
        else:
            assert(False)

        corrected_depths = intermediate_depths * intermediate_correction_factors
        if self.atm.wavelength_bins is None:
            return xp.array(intermediate_lambdas), xp.array(corrected_depths)

        if self.atm.method == "xsec" and self._xsec_bin_left_indices is not None and self._xsec_bin_right_indices is not None:
            left = self._xsec_bin_left_indices
            right = self._xsec_bin_right_indices
            prefix_l = xp.concatenate([xp.asarray([0.0]), xp.cumsum(intermediate_lambdas)])
            prefix_w = xp.concatenate([xp.asarray([0.0]), xp.cumsum(intermediate_stellar_spectrum)])
            prefix_dw = xp.concatenate([xp.asarray([0.0]), xp.cumsum(corrected_depths * intermediate_stellar_spectrum)])

            lambda_sums = prefix_l[right] - prefix_l[left]
            counts = right - left
            binned_wavelengths = lambda_sums / counts

            weight_sums = prefix_w[right] - prefix_w[left]
            weighted_depth_sums = prefix_dw[right] - prefix_dw[left]
            binned_depths = weighted_depth_sums / weight_sums
            return xp.asarray(binned_wavelengths), xp.asarray(binned_depths)

        binned_wavelengths = []
        binned_depths = []
        for start, end in self.atm.wavelength_bins:
            l = int(xp.cpu(xp.searchsorted(intermediate_lambdas, start)))
            r = int(xp.cpu(xp.searchsorted(intermediate_lambdas, end)))
            binned_wavelengths.append(xp.mean(intermediate_lambdas[l:r]))
            binned_depth = xp.average(
                corrected_depths[l:r],
                weights=intermediate_stellar_spectrum[l:r])
            binned_depths.append(binned_depth)

        return xp.asarray(binned_wavelengths), xp.asarray(binned_depths)

    def _validate_params(self, T, logZ, CO_ratio, cloudtop_pressure):
        T_profile = xp.ones(NUM_LAYERS) * T
        self.atm._validate_params(T, logZ, CO_ratio, cloudtop_pressure)
        
    
    def compute_depths(self, t_p_profile, star_radius, planet_mass, planet_radius,
                       logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       log_SO2=None, log_CH4=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=xp.inf, custom_abundances=None,
                       custom_T_profile=None, custom_P_profile=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       full_output=False, min_abundance=1e-99, min_cross_sec=1e-99, stellar_blackbody=False, zero_opacities=[]):
        '''
        Computes transit depths at a range of wavelengths, assuming an
        isothermal atmosphere.  To choose bins, call change_wavelength_bins().

        Parameters
        ----------
        t_p_profile : float
            Profile object
        star_radius : float
            Radius of the star
        planet_mass : float
            Mass of the planet, in kg
        planet_radius : float
            Radius of the planet at 100,000 Pa. Must be in metres.
        logZ : float
            Base-10 logarithm of the metallicity, in solar units
        CO_ratio : float, optional
            C/O atomic ratio in the atmosphere.  The solar value is 0.53.
        CH4_mult : float
            Multiple applied to equilibrium CH4 abundance, for methane depletion
        add_gas_absorption: float, optional
            Whether gas absorption is accounted for
        add_H_minus_absorption: float, optional
            Whether H- bound-free and free-free absorption is added in
        add_scattering : bool, optional
            whether Rayleigh scattering is taken into account
        scattering_factor : float, optional
            if `add_scattering` is True, make scattering this many
            times as strong. If `scattering_slope` is 4, corresponding to
            Rayleigh scattering, the absorption coefficients are simply
            multiplied by `scattering_factor`. If slope is not 4,
            `scattering_factor` is defined such that the absorption coefficient
            is that many times as strong as Rayleigh scattering at
            `scattering_ref_wavelength`.
        scattering_slope : float, optional
            Wavelength dependence of scattering, with 4 being Rayleigh.
        scattering_ref_wavelength : float, optional
            Scattering is `scattering_factor` as strong as Rayleigh at this
            wavelength, expressed in metres.
        add_collisional_absorption : float, optional
            Whether collisionally induced absorption is taken into account
        cloudtop_pressure : float, optional
            Pressure level (in Pa) below which light cannot penetrate.
            Use xp.inf for a cloudless atmosphere.
        custom_abundances : str or dict of xp.ndarray, optional
            If specified, overrides `logZ` and `CO_ratio`.  Can specify a
            filename, in which case the abundances are read from a file in the
            format of the EOS/ files.  These are identical to ExoTransmit's
            EOS files.  It is also possible, though highly discouraged, to
            specify a dictionary mapping species names to numpy arrays, so that
            custom_abundances['Na'][3,4] would mean the fractional number
            abundance of Na at a temperature of self.T_grid[3] and pressure of
            self.P_grid[4].
        custom_T_profile : array-like, optional
            If specified and custom_P_profile is also specified, divides the
            atmosphere into user-specified P/T points, instead of assuming an
            isothermal atmosphere with T = `temperature`.
        custom_P_profile : array-like, optional
            Must be specified along with `custom_T_profile` to use a custom
            P/T profile.  Pressures must be in Pa.
        T_star : float, optional
            Effective temperature of the star.  If you specify this and
            use wavelength binning, the wavelength binning becomes
            more accurate.
        T_spot : float, optional
            Effective temperature of the star spots. This can be used to make
            wavelength dependent correction to the observed transit depths.
        spot_cov_frac : float, optional
            The spot covering fraction of the star by area. This can be used to
            make wavelength dependent correction to the transit depths.
        ri : complex, optional
            Complex refractive index n - ik (where k > 0) of the particles
            responsible for Mie scattering.  If provided, Mie scattering will
            be computed.  In that case, scattering_factor and scattering_slope
            must be set to 1 and 4 (the default values) respectively.
        frac_scale_height : float, optional
            The number density of Mie scattering particles is proportional to
            P^(1/frac_scale_height).  This is similar to, but a bit different
            from, saying that the scale height of the particles is
            frac_scale_height times that of the gas.
        number_density: float, optional
            The number density (in m^-3) of Mie scattering particles
        part_size : float, optional
            The mean radius of Mie scattering particles.  The distribution is
            assumed to be log-normal, with a standard deviation of part_size_std
        part_size_std : float, optional
            The geometric standard deviation of particle radii. We recommend
            leaving this at the default value of 0.5.
        P_quench : float, optional
            Quench pressure in Pa.
        stellar_blackbody : bool, optional
            Whether to use a PHOENIX model for the stellar spectrum, or a blackbody
        zero_opacities : list of strings                                                                                                                                                                   
            List of molecules to zero opacities for
        full_output : bool, optional
            If True, returns info_dict as a third return value.


        Raises
        ------
        ValueError
            Raised when invalid parameters are passed to the method

        Returns
        -------
        wavelengths : array of float
            Central wavelengths, in metres
        transit_depths : array of float
            Transit depths at `wavelengths`
        info_dict : dict
            Returned if full_output is True, containing intermediate quantities
            calculated by the method.  These are: absorption_coeff_atm, tau_los,
            stellar_spectrum, radii, P_profile, T_profile, mu_profile,
            atm_abundances, unbinned_depths, unbinned_wavelengths
       '''
        if custom_P_profile is not None:
            if custom_T_profile is None or len(
                    custom_P_profile) != len(custom_T_profile):
                raise ValueError("Must specify both custom_T_profile and "
                                 "custom_P_profile, and the two must have the"
                                 " same length")
            #if temperature is not None:
            #    raise ValueError(
            #        "Cannot specify both temperature and custom T profile")
            
            P_profile = custom_P_profile
            T_profile = custom_T_profile
        else:
            #P_profile = xp.logspace(
            #    xp.log10(self.atm.P_grid[0]),
            #    xp.log10(self.atm.P_grid[-1]),
            #    NUM_LAYERS)
            #T_profile = xp.ones(len(P_profile)) * temperature
            P_profile = t_p_profile.pressures
            T_profile = t_p_profile.temperatures 

        atm_info = self.atm.compute_params(
            star_radius, planet_mass, planet_radius, P_profile, T_profile,
            logZ, CO_ratio, CH4_mult, gases, vmrs,log_SO2, log_CH4, add_gas_absorption, add_H_minus_absorption,
            add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            T_star, T_spot, spot_cov_frac, ri, frac_scale_height,
            number_density, part_size, part_size_std, P_quench, zero_opacities=zero_opacities,
            include_atm_abundances=full_output)

        radii = atm_info["radii"]
        dr = atm_info["dr"]
        tau_los = get_line_of_sight_tau(atm_info["absorption_coeff_atm"],
                                        radii)
        absorption_fraction = 1 - xp.exp(-tau_los)

        transit_depths = (radii.min() / star_radius)**2 \
            + 2 / star_radius**2 * absorption_fraction.dot(radii[1:] * dr)
        
        #For correlated-k: transit_depths has n_gauss points for every wavelength; unbinned_depths
        #has 1 point for every wavelength
        if full_output:
            binned_wavelengths, binned_depths, binned_stellar_spectrum, unbinned_wavelengths, unbinned_depths, unbinned_stellar_spectrum, unbinned_correction_factors = self._get_binned_corrected_depths(transit_depths, T_star, T_spot, spot_cov_frac, stellar_blackbody)
        else:
            binned_wavelengths, binned_depths = self._get_binned_depths_only(
                transit_depths, T_star, T_spot, spot_cov_frac, stellar_blackbody)
        
        if full_output:
            atm_info["tau_los"] = xp.cpu(tau_los)
            atm_info["binned_stellar_spectrum"] = xp.cpu(binned_stellar_spectrum)
            atm_info["unbinned_wavelengths"] = xp.cpu(unbinned_wavelengths)
            atm_info["unbinned_depths"] = xp.cpu(unbinned_depths)
            atm_info["unbinned_stellar_spectrum"] = xp.cpu(unbinned_stellar_spectrum)
            atm_info["unbinned_correction_factors"] = xp.cpu(unbinned_correction_factors)
            atm_info["contrib"] = xp.cpu(absorption_fraction)
            
            for key in atm_info:
                if type(atm_info[key]) == dict:
                    for subkey in atm_info[key]:
                        atm_info[key][subkey] = xp.cpu(atm_info[key][subkey])
                else:
                    atm_info[key] = xp.cpu(atm_info[key])
            return xp.cpu(binned_wavelengths), xp.cpu(binned_depths), atm_info

        return xp.cpu(binned_wavelengths), xp.cpu(binned_depths), None

    def compute_depths_patchy(self, t_p_profile, star_radius, planet_mass, planet_radius,
                       cloud_cov_frac,
                       logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       log_SO2=None, log_CH4=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=xp.inf, custom_abundances=None,
                       custom_T_profile=None, custom_P_profile=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       full_output=False, min_abundance=1e-99, min_cross_sec=1e-99, stellar_blackbody=False, zero_opacities=[]):
        """
        Computes transit depths for a patchy cloud model.
        This method is an optimized version for patchy clouds. It first computes
        the clear-sky model to cache the gas absorption, then computes the cloudy
        model reusing the cached data, and finally combines them.
        """
        # Tell AtmosphereSolver to cache the gas absorption computed for the clear case
        self.atm.cache_gas_absorption = True
        self.atm.cache_scattering_base = True
        self.atm.cache_profile_quantities = True

        # First, compute clear atmosphere to populate cache
        wavelengths_clear, depths_clear, _ = self.compute_depths(
            t_p_profile, star_radius, planet_mass, planet_radius,
            logZ, CO_ratio, CH4_mult, gases, vmrs,log_SO2, log_CH4, add_gas_absorption, add_H_minus_absorption,
            True, 1, 4, scattering_ref_wavelength,
            add_collisional_absorption, xp.inf, custom_abundances,
            custom_T_profile, custom_P_profile,
            T_star, T_spot, spot_cov_frac,
            None, 1, 0, # ri, frac_scale_height, number_density for clear
            part_size, part_size_std, P_quench,
            False, # full_output=False, we don't need the info dict
            min_abundance, min_cross_sec, stellar_blackbody, zero_opacities)

        # Now compute cloudy atmosphere, which will reuse cache
        wavelengths_cloudy, depths_cloudy, info_cloudy = self.compute_depths(
            t_p_profile, star_radius, planet_mass, planet_radius,
            logZ, CO_ratio, CH4_mult, gases, vmrs, log_SO2, log_CH4, add_gas_absorption, add_H_minus_absorption,
            add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, cloudtop_pressure, custom_abundances,
            custom_T_profile, custom_P_profile,
            T_star, T_spot, spot_cov_frac, ri, frac_scale_height,
            number_density, part_size, part_size_std, P_quench,
            full_output, min_abundance, min_cross_sec, stellar_blackbody, zero_opacities)

        # Reset cache state
        self.atm.cache_gas_absorption = False
        self.atm.gas_absorption_cache = None
        self.atm.cache_scattering_base = False
        self.atm.scattering_base_cache = None
        self.atm.cache_profile_quantities = False
        self.atm.profile_quantities_cache = None

        final_depths = cloud_cov_frac * depths_cloudy + (1 - cloud_cov_frac) * depths_clear

        if full_output:
            return wavelengths_cloudy, final_depths, info_cloudy
        else:
            return wavelengths_cloudy, final_depths, None
        
        
