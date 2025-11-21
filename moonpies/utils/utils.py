"""
Utility functions for moonpies sim
"""

import gc
import numpy as np
from functools import _lru_cache_wrapper


def get_ice_thickness(global_ice_mass, cfg, coldtraps=True):
    """Return polar ice thickness [m] given globally delivered ice mass [kg].

    Assumes ice is uniformly distributed over polar coldtrap area and has a 
    constant ice density. Scale by amount of global ice that reaches poles.

    Parameters
    ----------
    global_ice_mass : np.ndarray
        Ice delivered globally [kg].
    cfg : moonpies.config.Cfg, optional
        cfg.ice_species : str
            Ice species {"H2O", "CO2"}.
        cfg.coldtrap_area_H2O, cfg.coldtrap_area_CO2 : float
            Area of cold traps [m^2].
        cfg.ballistic_hop_effcy : float
            Ballistic hop efficiency (fraction of ice that reaches poles).
        cfg.ice_density : float
            Ice density [kg/m^3].

    Returns
    -------
    ice_thickness : np.ndarray
        Polar ice thickness [m].
    """
    if coldtraps:
        if cfg.ice_species == "H2O":
            coldtrap_area = cfg.coldtrap_area_H2O
        elif cfg.ice_species == "CO2":
            coldtrap_area = cfg.coldtrap_area_CO2
        polar_ice_mass = global_ice_mass * cfg.ballistic_hop_effcy  # [kg]
        ice_volume = polar_ice_mass / cfg.ice_density  # [m^3]
        ice_thickness = ice_volume / coldtrap_area

    else:
        # uniform distribution of ice mass over area covered by map
        # map_area = (cfg.grdxsize * cfg.grdstep) * (cfg.grdysize * cfg.grdstep)
        # polar_ice_mass = global_ice_mass * (map_area) / cfg.sa_moon
        # ice_volume = polar_ice_mass / cfg.ice_density
        # ice_thickness = ice_volume / (map_area)
        # comes out to the same as a uniform distribution of ice mass over entire moon
        ice_thickness = (global_ice_mass) / (cfg.ice_density * cfg.sa_moon)
    
    return ice_thickness


def impactor_mass2water(impactor_mass, cfg):
    """
    Return water [kg] from impactor mass [kg] using assumptions of Cannon 2020:
        - 36% of impactors are C-type (Jedicke et al., 2018)
        - 2/3 of C-types are hydrated (Rivkin, 2012)
        - Hydrated impactors are 10% water by mass (Cannon et al., 2020)
        - 16% of asteroid mass retained on impact (Ong et al., 2010)
     for comets:
        - assume 50% of comet hydrated
        - 6.5% of comet mass retained on impact (Ong et al., 2010)
    """
    if cfg.is_comet:
        mass_h2o = (
            cfg.comet_hydrated_wt_pct * impactor_mass * cfg.comet_mass_retained
        )
    else:
        mass_h2o = (
            cfg.ctype_frac
            * cfg.ctype_hydrated
            * cfg.hydrated_wt_pct
            * impactor_mass
            * cfg.impact_mass_retained
        )
    return mass_h2o


# Thermal module
def specific_heat_capacity(T, cfg):
    """
    Return specific heat capacity [J/kg/K] at temperature T [K] for lunar
    regolith (Hayne et al., 2017).

    Parameters
    ----------
    T (num or array): temperature [K]
    cfg (Cfg): Config object with specific_heat_coeffs
    """
    c0, c1, c2, c3, c4 = cfg.specific_heat_coeffs
    return c0 + c1 * T + c2 * T**2 + c3 * T**3 + c4 * T**4

# Pre-compute grid functions
def get_coldtrap_dists(df, cfg):
    """Return great circle distance between all craters and coldtraps. 
    
    Returns a 2D array (rows: craters from df, cols: cold traps from 
    cfg.coldtrap_names, in order). Elements are NaN for distance from crater 
    to its own cold trap or distance further than ejecta_threshold radii.


    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    cfg : moonpies.config.Cfg, optional
        cfg.coldtrap_names : tuple of str
            Tuple of cold trap names.
        cfg.ejecta_threshold : float
            Continuous ejecta threshold distance [crater radii].
        cfg.basin_ej_threshold : float
            Continuous ejecta threshold distance for basins [basin radii].

    Returns
    -------
    numpy.ndarray
        2D array of distances from craters to cold traps (Ncrater, Ncoldtrap).
    """
    dist = np.zeros((len(df), len(cfg.coldtrap_names)), dtype=cfg.dtype)
    local_coords = True if 'x' in df.columns else False
    for i, row in df.iterrows():

        if row.isbasin:
            ej_threshold = cfg.basin_ej_threshold
        else:
            ej_threshold = cfg.ej_threshold
        if ej_threshold < 0:
            ej_threshold = np.inf

        if local_coords:
            src_x = row.x
            src_y = row.y
            src_rad = row.rad
        else:
            src_lon = row.lon
            src_lat = row.lat
            src_rad = row.rad

        for j, cname in enumerate(cfg.coldtrap_names):
            if row.cname == cname:
                dist[i, j] = np.nan
                continue

            if local_coords:
                dst_x = df[df.cname == cname].x.values
                dst_y = df[df.cname == cname].y.values
                d = np.sqrt((dst_x-src_x)**2 + (dst_y-src_y)**2)
            else:
                dst_lon = df[df.cname == cname].psr_lon.values
                dst_lat = df[df.cname == cname].psr_lat.values
                d = gc_dist(src_lon, src_lat, dst_lon, dst_lat)

            # Only keep distances outside crater radius and less than thresh
            if src_rad < d < ej_threshold * src_rad:
                dist[i, j] = d
    
    dist[dist <= 0] = np.nan  # Set zeros to NaN
    return dist

def ages2time(
    time_arr, ages, values, cfg, agg=np.nansum, fillval=0, dtype=None
):
    """Return values shaped like time_arr given corresponding ages.

    Rounds ages to nearest time in time_arr. If multiple values have the same 
    age, they are combined using agg. Times with no ages are set to fillval.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    ages : numpy.ndarray
        Age of each value [yr].
    values : numpy.ndarray
        Values to reshape (must be same length as ages).
    agg : callable, optional
        Function to combine multiple values of same age, by default np.nansum.
    fillval : int, optional
        Value to use as default (all times lacking a value), by default 0.
    dtype : data-type, optional
        Data type of output, by default same as values.
    cfg : moonpies.config.Cfg, optional
        cfg.rtol : float
            Relative tolerance for time comparison to avoid rounding error.

    Returns
    -------
    values_time : numpy.ndarray
        Values reshaped like time_arr.
    """
    if dtype is None:
        dtype = values.dtype
    # Remove ages and associated values not in time_arr
    tmin, tmax = minmax_rtol(time_arr, cfg.rtol)  # avoid rounding error
    isin = np.where((ages >= tmin) & (ages <= tmax))
    ages = ages[isin]
    values = values[isin]

    # Minimize distance between ages and times to handle rounding error
    time_idx = np.abs(time_arr[:, np.newaxis] - ages).argmin(axis=0)
    shape = [len(time_arr), *values.shape[1:]]
    values_time = np.full(shape, fillval, dtype=dtype)

    for i, t_idx in enumerate(time_idx):
        # Sum here in case more than one crater formed at t_idx
        values_time[t_idx] = agg([values_time[t_idx], values[i]], axis=0)
    return values_time


# Geospatial helpers
def get_grid_arrays(cfg, half=False):
    """
    Return sparse meshgrid (grdy, grdx) from sizes [m], steps [m] and dtype.

    Grid is centered on South pole with y in (ysize, -ysize) and c in
    (-xsize, xsize) therefore total gridsize is (2*xsize)*(2*ysize).
    """
    ysize, ystep = cfg.grdysize, cfg.grdstep
    xsize, xstep = cfg.grdxsize, cfg.grdstep
    if half:
        grdy, grdx = np.meshgrid(
            np.arange(ysize, 0, -ystep, dtype=cfg.dtype),
            np.arange(0, xsize, xstep, dtype=cfg.dtype),
            sparse=True,
            indexing='ij'
        )
    else:
        grdy, grdx = np.meshgrid(
            np.arange(ysize, -ysize, -ystep, dtype=cfg.dtype),
            np.arange(-xsize, xsize, xstep, dtype=cfg.dtype),
            sparse=True,
            indexing="ij",
        )
    return grdy, grdx


def latlon2xy(lat, lon, rp=1737.4e3):
    """
    Return (x, y) [m] South Polar stereo coords from (lat, lon) [deg].

    Parameters
    ----------
    lat (num or arr): Latitude(s) [deg]
    lon (num or arr): Longitude(s) [deg]
    rp (num): Radius of the planet or moon [m]

    Returns
    -------
    x (num or arr): South Pole stereo x coordinate(s) [m]
    y (num or arr): South Pole stereo y coordinate(s) [m]
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = rp * np.cos(lat) * np.sin(lon)
    y = rp * np.cos(lat) * np.cos(lon)
    return x, y


def xy2latlon(x, y, rp=1737.4e3):
    """
    Return (lat, lon) [deg] from South Polar stereo coords (x, y) [m].

    Parameters
    ----------
    x (num or arr): South Pole stereo x coordinate(s) [m]
    y (num or arr): South Pole stereo y coordinate(s) [m]
    rp (num): Radius of the planet or moon [m]

    Returns
    -------
    lat (num or arr): Latitude(s) [deg]
    lon (num or arr): Longitude(s) [deg]
    """
    z = np.sqrt(rp**2 - x**2 - y**2)
    lat = np.rad2deg(-np.arcsin(z / rp))
    lon = np.rad2deg(np.arctan2(x, y))
    return lat, lon


def gc_dist(lon1, lat1, lon2, lat2, rp=1737.4e3):
    """
    Return great circle distance [m] from (lon1, lat1) - (lon2, lat2) [deg].

    Uses the Haversine formula adapted from C. Veness
    https://www.movable-type.co.uk/scripts/latlong.html

    Parameters
    ----------
    lon1 (num or arr): Longitude [deg] of start point
    lat1 (num or arr): Latitude [deg] of start point
    lon2 (num or arr): Longitude [deg] of end point
    lat2 (num or arr): Latitude [deg] of end point
    rp (num): Radius of the planet or moon [m]

    Returns
    -------
    gc_dist (num or arr): Great circle distance(s) in meters [m]
    """
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    sin2_dlon = np.sin((lon2 - lon1) / 2) ** 2
    sin2_dlat = np.sin((lat2 - lat1) / 2) ** 2
    a = sin2_dlat + np.cos(lat1) * np.cos(lat2) * sin2_dlon
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = rp * c
    return dist


# General helpers
def probabilistic_round(x, rng=None):
    """
    Randomly round float x up or down weighted by its distance to x + 1.

    E.g. 6.1 will round down ~90% of the time and round up ~10% of the time
    such that over many trials the expected value is 6.1.

    Modified from C. Locke https://stackoverflow.com/a/40921597/8742181

    Parameters
    ----------
    x (float): Any float (works on positive and negative values).

    Returns
    -------
    x_rounded (int): Either floor(x) or ceil(x), rounded probabalistically
    """
    rng = np.random.default_rng(rng)
    x = np.atleast_1d(x)
    random_offset = rng.random(x.shape)
    x_rounded = np.floor(x + random_offset).astype(int)
    return x_rounded


def round_to_ts(values, timestep):
    """Return values rounded to nearest timestep."""
    return np.around(values / timestep) * timestep


def rtol(num, tol):
    """Return relative tolerance where value is equal to num at tol."""
    return np.abs(num) * tol


def minmax_rtol(arr, tol):
    """Return min and max of arr -/+ relative tolerance tol."""
    amin = np.min(arr)
    amax = np.max(arr)
    return amin - rtol(amin, tol), amax + rtol(amax, tol)


def diam2vol(diameter):
    """Return volume of sphere [m^3] given diameter [m]."""
    return (4 / 3) * np.pi * (diameter / 2) ** 3


def m2km(x):
    """Convert meters to kilometers."""
    return x / 1000


def km2m(x):
    """Convert kilometers to meters."""
    return x * 1000


def vprint(cfg, *args, **kwargs):
    """Print if verbose."""
    if cfg.verbose:
        print(*args, **kwargs)


def clear_cache():
    """Reset lru_cache."""
    # All objects collected
    for obj in gc.get_objects():
        if isinstance(obj, _lru_cache_wrapper):
            obj.cache_clear()
