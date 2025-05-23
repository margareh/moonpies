"""
Ballistic sedimentation and hopping module
"""

import numpy as np
from moonpies.utils.load_data import read_ballistic_melt_frac
from moonpies.utils.utils import ages2time
from moonpies.utils.save_output import get_ejecta_thickness_time

# Ballistic sedimentation module
def get_melt_frac(ejecta_temps, mixing_ratios, cfg):
    """Return fraction of ice melted at given ejecta_temps and vol_fracs.

    Parameters
    ----------
    ejecta_temps : numpy.ndarray
        Ejecta temperatures [K].
    mixing_ratios : numpy.ndarray
        Mixing ratios (target:ejecta).
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.

    Returns
    -------
    melt_frac : numpy.ndarray
        Fraction of ice melted at each ejecta_temp and mixing ratio.

    See Also
    --------
    read_ballistic_melt_frac
    """
    def insert_unique_in_range(src, target):
        """Return sorted target with unique values in src inserted."""
        uniq = np.unique(src[~np.isnan(src)])
        # Ensure in range of target
        uniq = uniq[(target.min() < uniq) & (uniq < target.max())]
        arr = np.sort(np.unique(np.concatenate([target, uniq])))
        return arr

    mdf = read_ballistic_melt_frac(True, cfg)
    temps = insert_unique_in_range(ejecta_temps, mdf.columns.to_numpy())
    mrs = insert_unique_in_range(mixing_ratios, mdf.index.to_numpy())
    mdf = mdf.reindex(index=mrs, columns=temps)
    minterp = mdf.interpolate(axis=0).interpolate(axis=1)

    # Interpolate melt_frac at each non-nan ejecta_temp, mixing_ratio
    inds = np.argwhere(~np.isnan(mixing_ratios))
    melt_frac = np.zeros_like(mixing_ratios)
    for i, j in inds:
        melt_frac[i, j] = minterp.loc[mixing_ratios[i, j], ejecta_temps[i, j]]
    return melt_frac


def kinetic_energy(mass, velocity):
    """Return kinetic energy [J] given mass [kg] and velocity [m/s]."""
    return 0.5 * mass * velocity**2


def ejecta_temp(df, cfg, shape=None):
    """Return ejecta temperature [K].

    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    shape : tuple, optional
        Shape of output, by default (len(df),)
    cfg : moonpies.config.Cfg, optional
        cfg.polar_ejecta_temp_init : float
            Initial polar ejecta temperature [K].
        cfg.basin_ejecta_temp_warm : bool
            If True, use warm ejecta temperatures for basins, otherwise cold.
        cfg.basin_ejecta_temp_init_warm : float
            Initial warm basin ejecta temperature [K].
        cfg.basin_ejecta_temp_init_cold : float
            Initial cold basin ejecta temperature [K].

    Returns
    -------
    ejecta_temp : numpy.ndarray
        Ejecta temperature [K] for each impact in df.
    """
    if shape is None:
        shape = (len(df),)
    ejecta_t = np.ones(shape) * cfg.polar_ejecta_temp_init

    if cfg.basin_ejecta_temp_warm:
        basin_t = cfg.basin_ejecta_temp_init_warm
    else:
        basin_t = cfg.basin_ejecta_temp_init_cold
    ejecta_t[df.isbasin] = basin_t
    return ejecta_t


def ballistic_velocity(dist, cfg):
    """Return ballistic velocity given distance of travel on a sphere [1]_.

    Parameters
    ----------
    dist : numpy.ndarray
        Distance of travel [m].
    cfg : moonpies.config.Cfg, optional
        cfg.grav_moon : float
            Lunar gravity [m/s^2].
        cfg.rad_moon : float
            Lunar radius [m].
        cfg.impact_angle : float
            Impact angle [deg].

    Returns
    -------
    velocity : numpy.ndarray
        Ballistic velocity [m/s].

    References
    ----------
    .. [1] Vickery, A. (1986). "Size-velocity distribution of large ejecta 
       fragments." Icarus, 67(2), 224-236. https://doi.org/10/bjjk93

    """    
    theta = np.radians(cfg.impact_angle)
    g = cfg.grav_moon
    rp = cfg.rad_moon
    tan_phi = np.tan(dist / (2 * rp))
    num = g * rp * tan_phi
    denom = (np.sin(theta) * np.cos(theta)) + (np.cos(theta) ** 2 * tan_phi)
    return np.sqrt(num / denom)


def get_bsed_depth(time_arr, df, ej_dists, cfg):
    """Return ballistic sedimentation depth over time [1]_.

    If more than one ballistic sedimentation event occurs in a timestep, take
    the maximum depth and melt fraction. TODO: make sequential.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater DataFrame.
    ej_dists : numpy.ndarray
        2D array of ejecta distances [m] (Ncrater, Ncoldtrap).
    cfg : moonpies.config.Cfg, optional
        

    Returns
    -------
    depths : numpy.ndarray
        2D array of ballistic sedimentation depths [m] (Ntime, Ncoldtrap).
    melt_fracs : numpy.ndarray
        2D array of melt fractions (Ntime, Ncoldtrap).

    See Also
    --------
    get_mixing_ratio_oberbeck, ejecta_temp, get_melt_frac

    References
    ----------
    .. [1] Zhang, X., Xie, M., & Xiao, Z. (2021). "Thickness of orthopyroxene-
       rich materials of ejecta deposits from the south pole-Aitken basin." 
       Icarus, 358, 114214. https://doi.org/10/gmfc53
    """
    if not cfg.ballistic_sed:
        depths = np.zeros((len(time_arr), len(cfg.coldtrap_names)), cfg.dtype)
        fracs = np.zeros((len(time_arr), len(cfg.coldtrap_names)), cfg.dtype)
        return depths, fracs

    # Get distance, mixing_ratio, volume_frac for each crater to each coldtrap
    mixing_ratio = get_mixing_ratio_oberbeck(ej_dists, cfg)
    ej_temp = ejecta_temp(df, mixing_ratio.shape, cfg)
    melt_frac = get_melt_frac(ej_temp, mixing_ratio, cfg)

    # Convert to time array shape: (Ncrater, Ncoldtrap) -> (Ntime, Ncoldtrap)
    ej_thick_t, _ = get_ejecta_thickness_time(time_arr, df, ej_dists, cfg)
    ages = df.age.values
    mixing_ratio_t = ages2time(time_arr, ages, mixing_ratio, cfg, np.nanmax, 0)
    bsed_depths_t = ej_thick_t * mixing_ratio_t  # Petro and Pieters (2004)
    melt_frac_t = ages2time(time_arr, ages, melt_frac, cfg, np.nanmax, 0)
    melt_frac_t *= cfg.ballistic_sed_frac_lost  # Scale by fraction lost from column (default 100%)
    return bsed_depths_t, melt_frac_t


def get_mixing_ratio_oberbeck(ej_distances, cfg):
    """Return mixing ratio of target to ejecta material [1]_.

    If cfg.mixing_ratio_petro, use the adjustment for large mr from [2].

    Parameters
    ----------
    ej_distances : numpy.ndarray
        2D array of ejecta distances [m] (Ncrater, Ncoldtrap).
    cfg : moonpies.config.Cfg, optional
        cfg.mixing_ratio_a : float
            Mixing ratio factor a [1].
        cfg.mixing_ratio_b : float
            Mixing ratio exponent b [2].
        cfg.mixing_ratio_petro : bool
            If True, adjust large mixing ratios using [2]

    Returns
    -------
    mixing_ratio : numpy.ndarray
        2D array of mixing ratios (target:ejecta) (Ncrater, Ncoldtrap).

    References
    ----------
    .. [1] Oberbeck, V. R. (1975). "The role of ballistic erosion and 
       sedimentation in lunar stratigraphy." Reviews of Geophysics, 13(2), 
       337-362. https://doi.org/10/bcxqfb
    .. [2] Petro, N. E., & Pieters, C. M. (2006). "Modeling the provenance of 
       the Apollo 16 regolith." Journal of Geophysical Research, 111(E9), 
       E09005. https://doi.org/10/dbwzmv
    """
    dist = ej_distances * 1e-3  # [m -> km]
    mr = cfg.mixing_ratio_a * dist**cfg.mixing_ratio_b
    if cfg.mixing_ratio_petro:
        mrv = np.atleast_1d(mr)
        mrv[mrv > 5] = 0.5 * mrv[mrv > 5] + 2.5
        mr = mrv if mrv.ndim > 0 else float(mrv)
    return mr
