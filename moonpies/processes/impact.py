"""
Impact emplacement and gardening module
"""

import numpy as np

from moonpies import config

from moonpies.utils.load_data import read_ballistic_hop_csv
from moonpies.utils.utils import get_ice_thickness, probabilistic_round, impactor_mass2water, diam2vol, ages2time
from moonpies.utils.rv import _rng, get_random_hydrated_craters, get_random_impactor_speeds

from moonpies.models.sfd import get_small_impactor_pop, get_crater_pop, impact_flux_scaling
from moonpies.models.impactor import diam2len

# Impact gardening module (remove ice by impact overturn)
def remove_ice_overturn(ice_col, ej_col, t, depth, cfg):
    """Return ice_col gardened to depth below time t.

    Parameters
    ----------
    ice_col : numpy.ndarray
        Ice column [m].
    ej_col : numpy.ndarray
        Ejecta column [m].
    t : int
        Time index.
    depth : float
        Impact gardening depth [m].
    cfg : moonpies.config.Cfg, optional
        cfg.impact_gardening_costello : bool
            If True, use Costello impact gardening model, else use Cannon.

    Returns
    -------
    ice_col : numpy.ndarray
        Ice column [m] with impact gardening applied.

    See Also
    --------
    garden_ice_column, erode_ice_cannon
    """    
    if cfg.impact_gardening_costello:
        ice_col = garden_ice_column(ice_col, ej_col, t, depth)
    else:
        ice_col = erode_ice_cannon(ice_col, ej_col, t, depth)
    return ice_col


def erode_ice_cannon(ice_col, ej_col, t, erosion_depth=0.1, ej_shield=0.4):
    """Return ice_col gardened to erosion_depth below time t [1]_.

    Translated directly from Cannon et al. (2020) MATLAB code [1].

    Parameters
    ----------
    ice_col : numpy.ndarray
        Ice column [m].
    ej_col : numpy.ndarray
        Ejecta column [m].
    t : int
        Time index.
    erosion_depth : float, optional
        Erosion depth [m], by default 0.1
    ej_shield : float, optional
        Ejecta shield thickness [m], by default 0.4

    Returns
    -------
    ice_col : numpy.ndarray
        Ice column [m] with impact gardening applied.
    
    References
    ----------
    .. [1] Cannon, K. M., Deutsch, A. N., Head, J. W., & Britt, D. T. (2020). 
       "Stratigraphy of Ice and Ejecta Deposits at the Lunar Poles."
       Geophysical Research Letters, 47(21). https://doi.org/10/gksdcc

    """    
    # Possible bug: erosion base never updated for adjacent ejecta layers
    # Erosion base is most recent time when ejecta column was > ejecta_shield
    erosion_base = -1
    erosion_base_idx = np.where(ej_col[: t + 1] > ej_shield)[0]
    if len(erosion_base_idx) > 0:
        erosion_base = erosion_base_idx[-1]

    # Garden from top of ice column until ice_to_erode amount is removed
    layer = t
    while erosion_depth > 0 and layer >= 0:
        # Possible bug: t > erosion_base should be layer > erosion_base.
        # - loop doesn't end if we reach erosion base while eroding
        # - loop only ends in else block if we started at erosion_base
        if t > erosion_base:
            ice_in_layer = ice_col[layer]
            if ice_in_layer >= erosion_depth:
                ice_col[layer] -= erosion_depth
            else:
                ice_col[layer] = 0
            erosion_depth -= ice_in_layer
            layer -= 1
        else:
            break
    return ice_col


def garden_ice_column(ice_column, ejecta_column, t, depth, eff=1):
    """Return ice_column gardened to depth, but preserved by ejecta_column.

    Ejecta deposited on last timestep preserves ice. Loop through ice_col and
    ejecta_col until overturn_depth and remove all ice that is encountered.

    Parameters
    ----------
    ice_column : numpy.ndarray
        Ice column [m].
    ejecta_column : numpy.ndarray
        Ejecta column [m].
    t : int
        Time index.
    depth : float
        Impact gardening depth [m].
    eff : float, optional
        Fraction to remove from each layer in [0, 1], by default 1 (all).

    Returns
    -------
    ice_column : numpy.ndarray
        Ice column [m] with impact gardening applied.
    """
    # Travese ice and ejecta column from t down, removing ice, skipping ejecta
    # Loop until we hit the bottom or have gone down depth meters
    # - If ejecta[t] > depth, no ice is removed.
    # Double i so i//2 is current index to garden (odd: ejecta, even: ice)
    i = (2 * t) + 1
    d = 0  # current depth
    while i >= 0 and d < depth:  # and < 2 * len(ice_column):
        if i % 2:
            # Odd i (ejecta): do nothing, add ejecta layer to depth, d
            d += ejecta_column[i // 2]
        else:
            # Even i (ice): remove ice*eff from layer
            removed = ice_column[i // 2] * eff

            # Removing more ice than depth, only remove enough to reach depth
            if (d + removed) > depth:
                removed = depth - d
            ice_column[i // 2] -= removed
            d += ice_column[i // 2]  # Count all ice in layer towards depth
        i -= 1
    return ice_column


# Impact gardening module (Costello et al. 2018, 2020)
def overturn_depth_time(time_arr, cfg):
    """Return overturn depth [m] at each time.


    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.impact_gardening_costello : bool
            If True, get overturn over time, else use cfg.ice_erosion_rate.
        cfg.ice_erosion_rate : float
            Constant ice erosion rate [m/10 Ma].
        cfg.timestep : float
            Timestep [yr] to scale cfg.ice_erosion_rate.

    Returns
    -------
    overturn_depth : numpy.ndarray
        Overturn depth [m] at each time.

    See Also
    --------
    overturn_depth_costello_time
    """
    if cfg.impact_gardening_costello:
        overturn_t = overturn_depth_costello_time(time_arr, cfg)
    else:
        # Cannon mode assume ice_erosion_rate 0.1 m / Ma gardening at all times
        t_scaling = cfg.timestep / 1e7  # scale from 10 Ma rate
        overturn_t = cfg.ice_erosion_rate * t_scaling * np.ones_like(time_arr)
    return overturn_t


def overturn_depth_costello_time(time_arr, cfg):
    """Return regolith overturn depth at each time [1]_.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.overturn_ancient_t0 : float
            Time when ancient overturn rate becomes present rate [yr].
        cfg.overturn_depth_present : float
            Present-day overturn depth per cfg.timestep [m].
        cfg.overturn_ancient_slope : float
            Slope of ancient overturn depth [m/yr].

    Returns
    -------
    overturn_depth : numpy.ndarray
        Overturn depth [m] at each time.
    
    References
    ----------
    .. [1] Costello, E. S., Ghent, R. R., Hirabayashi, M., & Lucey, P. G. 
       (2020). "Impact Gardening as a Constraint on the Age, Source, and 
       Evolution of Ice on Mercury and the Moon." Journal of Geophysical 
       Research: Planets, 125(3). https://doi.org/10/gk56kr

    """    
    t_anc = cfg.overturn_ancient_t0
    depth = np.ones_like(time_arr) * cfg.overturn_depth_present
    d_scaling = cfg.overturn_ancient_slope * (time_arr - t_anc) + 1
    depth[time_arr > t_anc] *= d_scaling[time_arr > t_anc]
    return depth


# Impact-delivered ice module
def get_impact_ice(time_arr, df, cfg, rng=None):
    """Return polar ice thickness [m] from global impacts at each time.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater dataframe.
    cfg : moonpies.config.Cfg, optional
        cfg.impact_ice_basins : bool
            If True, add basin ice contributions
    rng : int or numpy.random.Generator, optional
        Random number generator or seed.

    Returns
    -------
    ice_time : numpy.ndarray
        Polar ice thickness [m] at each time.
    """
    impact_ice_t = np.zeros_like(time_arr)
    impact_ice_t += get_micrometeorite_ice(time_arr, cfg)
    impact_ice_t += get_small_impactor_ice(time_arr, cfg)
    impact_ice_t += get_small_simple_crater_ice(time_arr, cfg)
    impact_ice_t += get_large_simple_crater_ice(time_arr, cfg, rng)
    impact_ice_t += get_complex_crater_ice(time_arr, cfg, rng)
    impact_ice_basins_t = get_basin_ice(time_arr, df, cfg, rng)
    if cfg.impact_ice_basins:
        # get_basin_ice is run every time for repro, but only add if needed
        impact_ice_t += impact_ice_basins_t
    return impact_ice_t


def get_comet_cfg(cfg):
    """Return config with impact parameters updated to comet values.

    Sets cfg.is_comet, cfg.hydrated_wt_pct, cfg.impactor_density, 
    cfg.impact_mass_retained to comet counterparts.

    Parameters
    ----------
    cfg : moonpies.config.Cfg
        MoonPIES config object.

    Returns
    -------
    comet_cfg : moonpies.config.Cfg
        MoonPIES config object with comet parameters set.
    """    
    # TODO: Explicitly handle comets instead of cludge with impact cfg params.
    # Define comet_cfg with comet parameters to supply to get_impact_ice
    comet_cfg_dict = cfg.to_dict()
    comet_cfg_dict["is_comet"] = True  # Use comet impact speeds
    comet_cfg_dict["hydrated_wt_pct"] = cfg.comet_hydrated_wt_pct
    comet_cfg_dict["impactor_density"] = cfg.comet_density
    comet_cfg_dict["impact_mass_retained"] = cfg.comet_mass_retained
    return config.from_dict(comet_cfg_dict)


def get_impact_ice_comet(time_arr, df, cfg, rng=None):
    """Return polar ice thickness [m] from comet impacts at each time.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater dataframe.
    cfg : moonpies.config.Cfg
        MoonPIES config object.
    rng : int or numpy.random.Generator, optional
        Random number generator or seed.

    Returns
    -------
    ice_time : numpy.ndarray
        Comet polar ice thickness [m] at each time.
    """    
    # Repeat get_impact_ice with comet cfg
    comet_cfg = get_comet_cfg(cfg)
    comet_ice_t = get_impact_ice(time_arr, df, comet_cfg, rng)
    return comet_ice_t


def get_micrometeorite_ice(time_arr, cfg):
    """
    Return ice thickness [m] delivered to pole due to micrometeorites vs time.

    Returns
    -------
    mm_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    mm_ice_mass = ice_micrometeorites(time_arr, cfg)
    mm_ice_t = get_ice_thickness(mm_ice_mass, cfg)
    return mm_ice_t


def get_small_impactor_ice(time_arr, cfg):
    """
    Return ice thickness [m] delivered to pole due to small impactors vs time.

    Returns
    -------
    si_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    impactor_diams, impactors = get_small_impactor_pop(time_arr, cfg)
    si_ice_mass = ice_small_impactors(impactor_diams, impactors, cfg)
    si_ice_t = get_ice_thickness(si_ice_mass, cfg)
    return si_ice_t


def get_small_simple_crater_ice(time_arr, cfg):
    """
    Return ice thickness [m] delivered to pole by small simple crater impacts.

    Returns
    -------
    ssc_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    crater_diams, n_craters_t, sfd_prob = get_crater_pop(time_arr, "c", cfg)
    n_craters = n_craters_t * sfd_prob
    ssc_ice_mass = ice_small_craters(crater_diams, n_craters, "c", cfg)
    ssc_ice_t = get_ice_thickness(ssc_ice_mass, cfg)
    return ssc_ice_t


def get_large_simple_crater_ice(time_arr, cfg, rng=None):
    """
    Return ice thickness [m] delivered to pole by large simple crater impacts.

    Returns
    -------
    lsc_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    lsc_ice_mass = get_ice_stochastic(time_arr, "d", cfg, rng)
    lsc_ice_t = get_ice_thickness(lsc_ice_mass, cfg)
    return lsc_ice_t


def get_complex_crater_ice(time_arr, cfg, rng=None):
    """
    Return ice thickness [m] delivered to pole by complex crater impacts.

    Returns
    -------
    cc_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    cc_ice_mass = get_ice_stochastic(time_arr, "e", cfg, rng)
    cc_ice_t = get_ice_thickness(cc_ice_mass, cfg)
    return cc_ice_t


def get_basin_ice(time_arr, df, cfg, rng=None):
    """
    Return ice thickness [m] delivered to pole by basin impacts vs time.

    Returns
    -------
    b_ice_t (arr): Ice thickness [m] delivered to pole at each time.
    """
    b_ice_mass = ice_basins(df, time_arr, cfg, rng)
    b_ice_t = get_ice_thickness(b_ice_mass, cfg)
    return b_ice_t


def get_ice_stochastic(time_arr, regime, cfg, rng=None):
    """
    Return ice mass [kg] delivered to pole at each time from stochastic 
    delivery by crater-forming impactors.

    Parameters
    ----------
    time_arr (arr): Array of times [yr] to calculate ice thickness at.
    regime (str): Regime of ice production.
    cfg (Cfg): Config object.
    rng (int or numpy.random.Generator): Seed or random number generator.

    Returns
    -------
    ice_t (arr): Ice mass [kg] delivered to pole at each time.
    """
    rng = _rng(rng)
    diams, num_craters_t, sfd_prob = get_crater_pop(time_arr, regime, cfg)

    # Round to integer number of craters at each time
    num_craters_t = probabilistic_round(num_craters_t, rng=rng)

    # Get ice thickness for each time, randomizing crater diam and impact speed
    ice_mass_t = np.zeros_like(time_arr)
    for i, num_craters in enumerate(num_craters_t):
        # Randomly resample crater diams from sfd prob with replacement
        rand_diams = rng.choice(diams, num_craters, p=sfd_prob)

        # Randomly subset impacts to only hydrated
        hyd = get_random_hydrated_craters(len(rand_diams), cfg, rng)
        hyd_diams = rand_diams[hyd]

        speeds = get_random_impactor_speeds(len(hyd_diams), cfg, rng)
        ice_mass_t[i] = ice_large_craters(hyd_diams, speeds, regime, cfg)
    return ice_mass_t


def get_ballistic_hop_coldtraps(coldtraps, cfg):
    """
    Return ballistic hop efficiency as row vector of coldtraps [1,N].

    Parameters
    ----------
    coldtraps (list of str): Desired coldtraps.
    cfg (Cfg): Configuration object.

    Returns
    -------
    bhop (arr): Ballistic hop efficiency of coldtraps.
    """
    bhop = read_ballistic_hop_csv(cfg.bhop_csv_in)
    return bhop.loc[coldtraps].values.T


def ice_micrometeorites(time, cfg):
    """
    Return ice from micrometeorites (Regime A, Cannon 2020).

    Multiply total_mm_mass / yr by timestep and scale by assumed hydration %
    and scale by ancient flux relative to today.

    Unlike larger impactors, we DO NOT assume ctype composition and fraction of
    hydrated ctypes and also DO NOT scale by asteroid retention rate.
    TODO: are these reasonable assumptions?
    """
    # Scale by impact flux relative to today
    mm_mass_t = cfg.mm_mass_rate * impact_flux_scaling(time)

    if cfg.is_comet:
        # When comet micrometeorites (100% comets)
        ice_ret = cfg.comet_hydrated_wt_pct * cfg.comet_mass_retained
    elif cfg.impact_ice_comets:
        # When asteroid micrometeorites but comets included
        ice_ret = 0
    else:
        # When asteroid micrometeorites but comets excluded (100% asteroids)
        ice_ret = cfg.hydrated_wt_pct * cfg.impact_mass_retained
    mm_ice_t = mm_mass_t * ice_ret * cfg.timestep
    return mm_ice_t


def ice_small_impactors(diams, impactors_t, cfg):
    """
    Return ice mass [kg] from small impactors (Regime B, Cannon 2020) given
    impactor diams and number of impactors over time.
    """
    impactor_masses = diam2vol(diams) * cfg.impactor_density
    impactor_mass_t = np.sum(impactors_t * impactor_masses, axis=1)
    impactor_ice_t = impactor_mass2water(impactor_mass_t, cfg)
    return impactor_ice_t


def ice_small_craters(
    crater_diams,
    ncraters,
    regime,
    cfg
):
    """
    Return ice from simple craters, steep branch (Regime C, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, cfg.impact_speed_mean, regime, cfg)
    impactor_masses = diam2vol(impactor_diams) * cfg.impactor_density  # [kg]
    total_impactor_mass = np.sum(impactor_masses * ncraters, axis=1)
    total_impactor_water = impactor_mass2water(total_impactor_mass, cfg)
    return total_impactor_water


def ice_large_craters(crater_diams, impactor_speeds, regime, cfg):
    """
    Return ice from simple/complex craters, shallow branch of sfd
    (Regime D-E, Cannon 2020).
    """
    impactor_diams = diam2len(crater_diams, impactor_speeds, regime, cfg)
    impactor_masses = diam2vol(impactor_diams) * cfg.impactor_density  # [kg]

    # Find ice mass assuming hydration wt% and retention based on speed
    ice_retention = ice_retention_factor(impactor_speeds, cfg)
    ice_masses = impactor_masses * cfg.hydrated_wt_pct * ice_retention
    return np.sum(ice_masses)


def ice_basins(df, time_arr, cfg, rng=None):
    """Return ice mass [kg] from basin impacts vs time."""
    # Get basin ice from hydrated ctypes and cometary basins
    itype = "hyd_ctype"  # icy_impactor flag (see get_random_icy_basins)
    if cfg.is_comet:
        itype = "comet"

    basin_ice_t = np.zeros_like(time_arr)

    bdf = df[df.isbasin & (df.icy_impactor == itype)].reset_index(drop=True)
    basin_diams = 2 * bdf.rad.values
    impactor_speeds = get_random_impactor_speeds(len(bdf), cfg, rng)

    # Get ice mass of each basin impactor
    ice_masses = np.zeros_like(basin_diams)
    for i, (diam, speed) in enumerate(zip(basin_diams, impactor_speeds)):
        ice_masses[i] = ice_large_craters(diam, speed, "f", cfg)

    # Insert each basin ice mass to its position in time_arr
    ages = bdf.age.values
    basin_ice_t = ages2time(time_arr, ages, ice_masses, cfg, np.sum, 0)
    return basin_ice_t


def ice_retention_factor(speeds, cfg):
    """
    Return ice retained in impact, given impactor speeds (Ong et al. 2010).

    For speeds < 10 km/s, retain 50% (Svetsov & Shuvalov 2015 via Cannon 2020).
    For speeds >= 10 km/s, use fit Fig 2 (Ong et al. 2010 via Cannon 2020)
    """
    speeds = speeds * 1e-3  # [m/s] -> [km/s]
    retained = np.ones_like(speeds)
    if cfg.mode == "cannon":
        # Cannon et al. (2020) ds02
        retained[speeds >= 10] = 36.26 * np.exp(-0.3464 * speeds[speeds >= 10])
    elif cfg.mode == "moonpies":
        # Fit to Fig 2. (Ong et al. 2010), negligible retention > 45 km/s
        retained[speeds >= 10] = 1.66e4 * speeds[speeds >= 10] ** -4.16
    if not cfg.is_comet:
        retained[speeds < 10] = 0.5  # Cap at 50% retained for asteroids
    retained[retained < 0] = 0
    return retained

