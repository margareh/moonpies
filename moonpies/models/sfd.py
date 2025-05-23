"""
Crater size-frequency distributions
"""

import numpy as np
from functools import lru_cache


# Crater/impactor size-frequency helpers
@lru_cache(6)
def neukum(diam, cfg):
    """
    Return number of craters per m^2 per yr at diam [m] (eqn. 2, Neukum 2001).

    Eqn 2 expects diam [km], returns N [km^-2 Ga^-1].

    Parameters
    ----------
    diam (float): Crater diameter [m].
    cfg (Cfg): Config object.
    """
    if cfg.neukum_pf_new:
        a_vals = cfg.neukum_pf_a_2001
    else:
        a_vals = cfg.neukum_pf_a_1983
    diam = diam * 1e-3  # [m] -> [km]
    j = np.arange(len(a_vals))
    ncraters = 10 ** np.sum(a_vals * np.log10(diam) ** j)  # [km^-2 Ga^-1]
    return ncraters * 1e-6 * 1e-9  # [km^-2 Ga^-1] -> [m^-2 yr^-1]


def get_impactors_brown(mindiam, maxdiam, timestep, cfg):
    """
    Return number of impactors per yr in range mindiam, maxdiam (Brown et al.
    2002) and scale by Earth-Moon impact ratio (Mazrouei et al. 2019).
    """
    c0, d0 = cfg.brown_c0, cfg.brown_d0
    n_impactors_gt_low = 10 ** (c0 - d0 * np.log10(mindiam))  # [yr^-1]
    n_impactors_gt_high = 10 ** (c0 - d0 * np.log10(maxdiam))  # [yr^-1]
    n_impactors_earth_yr = n_impactors_gt_low - n_impactors_gt_high
    n_impactors_moon = n_impactors_earth_yr * timestep / cfg.earth_moon_ratio
    if cfg.is_comet:
        n_impactors_moon = n_impactors_moon * cfg.comet_ast_frac
    elif cfg.impact_ice_comets:
        n_impactors_moon = n_impactors_moon * (1 - cfg.comet_ast_frac)
    return n_impactors_moon


def get_crater_pop(time_arr, regime, cfg):
    """
    Return crater population assuming continuous (non-stochastic, regime c).
    """
    mindiam, maxdiam, step, slope = cfg.impact_regimes[regime]
    diams, sfd_prob = get_crater_sfd(mindiam, maxdiam, step, slope, cfg.dtype)
    num_craters_t = num_craters_chronology(mindiam, maxdiam, time_arr, cfg)
    num_craters_t = np.atleast_1d(num_craters_t)[:, np.newaxis]  # make col vec
    if cfg.is_comet:
        num_craters_t = num_craters_t * cfg.comet_ast_frac
    elif cfg.impact_ice_comets:
        num_craters_t = num_craters_t * (1 - cfg.comet_ast_frac)
    return diams, num_craters_t, sfd_prob


def num_craters_chronology(mindiam, maxdiam, time, cfg):
    """
    Return number of craters [m^-2 yr^-1] mindiam and maxdiam at each time.
    """
    # Compute number of craters from neukum pf
    fmax = neukum(mindiam, cfg)
    fmin = neukum(maxdiam, cfg)
    count = (fmax - fmin) * cfg.sa_moon * cfg.timestep

    # Scale count by impact flux relative to present day flux
    num_craters = count * impact_flux_scaling(time)
    return num_craters


def get_small_impactor_pop(time_arr, cfg):
    """
    Return population of impactors and number in regime B.

    Use constants and eqn. 3 from Brown et al. (2002) to compute N craters.
    """
    min_d, max_d, step, sfd_slope = cfg.impact_regimes["b"]
    diams, sfd_prob = get_crater_sfd(min_d, max_d, step, sfd_slope, cfg.dtype)
    n_impactors = get_impactors_brown(min_d, max_d, cfg.timestep, cfg)

    # Scale n_impactors by historical impact flux, csfd shape: (NT, Ndiams)
    flux_scaling = impact_flux_scaling(time_arr)
    n_impactors_t = n_impactors * flux_scaling[:, None] * sfd_prob
    return diams, n_impactors_t


@lru_cache(4)
def get_crater_sfd(dmin, dmax, step, sfd_slope, dtype=None):
    """
    Return diam_array and sfd_prob. This func makes it easier to cache both.
    """
    diam_array = get_diam_array(dmin, dmax, step, dtype)
    sfd_prob = get_sfd_prob(diam_array, sfd_slope)
    return diam_array, sfd_prob


def get_sfd_prob(diams, sfd_slope):
    """
    Return size-frequency distribution probability given diams, sfd slope.
    """
    sfd = diams**sfd_slope
    return sfd / np.sum(sfd)


def get_diam_array(dmin, dmax, step, dtype=None):
    """Return array of diameters based on diameters from dmin, dmax, dstep."""
    n = int((dmax - dmin) / step)
    return np.linspace(dmin, dmax, n + 1, dtype=dtype)


def impact_flux_scaling(time):
    """
    Return the factor to scale impact flux by in the past vs. present day.

    Take ratio of historical and present day impact fluxes from Ivanov 2008).
    Parameters
    ----------
    time (num or arr): Time [year] before present.

    Returns
    -------
    scaling_factor (num): Impact flux scaling factor.
    """
    scaling_factor = impact_flux(time) / impact_flux(0)
    return scaling_factor


def impact_flux(time):
    """Return impact flux at time [yrs] (Derivative of eqn. 1, Ivanov 2008)."""
    time = time * 1e-9  # [yrs -> Ga]
    flux = 6.93 * 5.44e-14 * (np.exp(6.93 * time)) + 8.38e-4  # [n/Ga]
    return flux * 1e-9  # [Ga^-1 -> yrs^-1]
