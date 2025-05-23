"""
Volcanic emplacement module
"""

import numpy as np
from moonpies.utils.load_data import read_volcanic_species
from moonpies.utils.utils import get_ice_thickness, rtol


# Volcanic ice delivery module
def get_volcanic_ice(time_arr, cfg):
    """Return ice thickness [m] delivered to pole by volcanics at each time.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.volc_mode : str
            Volcanic ice mode {"NK", "Head"}.

    Returns
    -------
    ice_time : numpy.ndarray
        Thickness of volcanic ice delivered [m] at each time.

    See Also
    --------
    volcanic_ice_nk, volcanic_ice_head
    """
    if cfg.volc_mode == "NK":
        volc_ice_mass = volcanic_ice_nk(time_arr, cfg)
    elif cfg.volc_mode == "Head":
        volc_ice_mass = volcanic_ice_head(time_arr, cfg)
    else:
        raise ValueError(f"Invalid mode {cfg.volc_mode}.")

    volc_ice_t = get_ice_thickness(volc_ice_mass, cfg)
    return volc_ice_t


def volcanic_ice_nk(time_arr, cfg):
    """Return global ice mass [kg] deposited at each time using [1]_.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.volc_mode : str

    Returns
    -------
    ice_mass : numpy.ndarray
        Global volcanic ice mass [kg] deposited at each time.
    """    
    df_volc = read_volcanic_species(cfg)
    tmax = time_arr.max()
    df_volc = df_volc[df_volc.time < tmax + rtol(tmax, cfg.rtol)]

    rounded_time = np.rint(time_arr / cfg.timestep)
    rounded_ages = np.rint(df_volc.time.values / cfg.timestep)
    time_idx = np.searchsorted(-rounded_time, -rounded_ages)

    # Compute volc ice mass at each time in time_arr
    #   Divide each ice mass by time between timesteps in df_volc
    volc_ice_mass = np.zeros_like(time_arr)
    for i, t_idx in enumerate(time_idx[:-1]):
        # Sum here in case more than one crater formed at t_idx
        next_idx = time_idx[i + 1]
        dt = (time_arr[t_idx] - time_arr[next_idx]) / cfg.timestep
        volc_ice_mass[t_idx : next_idx + 1] = (
            df_volc.iloc[i][cfg.nk_species] / dt
        )
    return volc_ice_mass


def volcanic_ice_head(time_arr, cfg):
    """Return global ice [kg] deposited at each time using Head et al. (2020).

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.volc_total_vol : float
            Total volcanics volume [km^3].
        cfg.volc_magma_density : float
            Magma density [kg/m^3].
        cfg.volc_ice_ppm : float
            Volcanic volatile concentration [ppm].
        cfg.volc_early : tuple of float
            Early volcanism time period (start, end) [yr].
        cfg.volc_early_pct : float
            Percent of total volcanics delivered during volc_early.
        cfg.volc_late : tuple of float
            Late volcanism time period (start, end) [yr].
        cfg.volc_late_pct : float
            Percent of total volcanics delivered during volc_late.

    Returns
    -------
    ice_mass : numpy.ndarray
        Global volcanic ice mass [kg] deposited at each time.
    """
    # Global ice deposition (Head et al. 2020)
    volc_mass = cfg.volc_total_vol * cfg.volc_magma_density
    ice_total = volc_mass * cfg.volc_ice_ppm * 1e-6  # [ppm yr^-1 -> yr^-1]

    # Ice deposited per epoch
    ice_early = ice_total * cfg.volc_early_pct * cfg.timestep
    ice_late = ice_total * cfg.volc_late_pct * cfg.timestep

    # Set ice at times in each epoch
    volc_ice_mass = np.zeros_like(time_arr)
    emax, emin = cfg.volc_early
    lmax, lmin = cfg.volc_late
    volc_ice_mass[(time_arr <= emax) & (time_arr > emin)] = ice_early
    volc_ice_mass[(time_arr <= lmax) & (time_arr > lmin)] = ice_late
    return volc_ice_mass
