"""
Ballistic sedimentation and hopping module
"""

import copy
import numpy as np

from scipy.interpolate import bisplrep, bisplev

from moonpies.utils.load_data import read_ballistic_melt_frac
from moonpies.utils.utils import ages2time
from moonpies.models.impactor import final2transient


def get_ejecta_thickness(distance, radius, cfg):
    """Return ejecta thickness as a function of distance given crater radius.

    Parameters
    ----------
    distance : numpy.ndarray
        Distance from crater center [m].
    radius : float
        Crater radius [m].
    cfg : moonpies.config.Cfg, optional
        cfg.simple2complex : float
            Simple to complex transition diameter [m]
        cfg.complex2peakring : float
            Complex to peak ring basin transition diameter [m]

    Returns
    -------
    thickness : numpy.ndarray
        Ejecta thickness [m] at distance.

    See Also
    --------
    get_ej_thick_simple, get_ej_thick_complex, get_ej_thick_basin
    """

    dist_v = np.atleast_1d(distance)
    rad_v = np.broadcast_to(radius, dist_v.shape)
    thick = np.zeros_like(dist_v * rad_v)

    # Simple craters
    is_s = rad_v < cfg.simple2complex / 2
    thick[is_s] = get_ej_thick_simple(dist_v[is_s], rad_v[is_s], cfg)

    # Complex craters
    is_c = (cfg.simple2complex / 2 < rad_v) & (
        rad_v < cfg.complex2peakring / 2
    )
    thick[is_c] = get_ej_thick_complex(dist_v[is_c], rad_v[is_c], cfg)

    # Basins
    is_pr = rad_v >= cfg.complex2peakring / 2
    t_radius = final2transient(rad_v * 2, cfg) / 2
    thick[is_pr] = get_ej_thick_basin(dist_v[is_pr], t_radius[is_pr], cfg)
    thick[np.isnan(thick)] = 0
    if np.ndim(distance) == 0 and len(thick) == 1:
        thick = float(thick)  # Return scalar if passed a scalar
    return thick


def get_ej_thick_simple(distance, radius, cfg):
    """Return ejecta thickness with distance from a simple crater [1]_.

    Parameters
    ----------
    distance : numpy.ndarray
        Distance from crater center [m].
    radius : float
        Crater final radius [m].
    cfg : moonpies.config.Cfg, optional
        cfg.ej_thickness_exp : float
            Ejecta thickness exponent, default: -3.

    Returns
    -------
    thickness : numpy.ndarray
        Ejecta thickness [m] at distance.

    References
    ----------
    .. [1] Kring, D. A. (1995). "The dimensions of the Chicxulub impact crater
       and impact melt sheet." Journal of Geophysical Research, 100(E8),
       16979. https://doi.org/10/fphq8j

    """    
    return 0.04 * radius * (distance / radius) ** cfg.ej_thickness_exp


def get_ej_thick_complex(distance, radius, cfg):
    """Return ejecta thickness with distance from a complex crater [1]_.

    Parameters
    ----------
    distance : numpy.ndarray
        Distance from crater center [m].
    radius : float
        Crater final radius [m].
    cfg : moonpies.config.Cfg, optional
        cfg.ej_thickness_exp : float
            Ejecta thickness exponent, default: -3.

    Returns
    -------
    thickness : numpy.ndarray
        Ejecta thickness [m] at distance.

    References
    ----------
    .. [1] Kring, D. A. (1995). "The dimensions of the Chicxulub impact crater
       and impact melt sheet." Journal of Geophysical Research, 100(E8),
       16979. https://doi.org/10/fphq8j
    """
    return 0.14 * radius**0.74 * (distance / radius) ** cfg.ej_thickness_exp


def get_ej_thick_basin(distance, t_radius, cfg):
    """Return ejecta thickness with distance from a basin [1]_.

    Uses Pike (1974) and Haskin et al. (2003) correction for curvature as
    described in [1].

    Parameters
    ----------
    distance : numpy.ndarray
        Distance from crater center [m].
    t_radius : float
        Basin transient crater radius [m].
    cfg : moonpies.config.Cfg, optional
        cfg.ej_thickness_exp : float
            Ejecta thickness exponent, default: -3.
        cfg.rad_moon : float
            Lunar radius [m].
        cfg.grav_moon : float
            Lunar gravity [m/s^2].
        cfg.impact_angle : float
            Impact angle [deg].

    Returns
    -------
    thickness : numpy.ndarray
        Ejecta thickness [m] at distance.

    References
    ----------
    .. [1] Zhang, X., Xie, M., & Xiao, Z. (2021). "Thickness of orthopyroxene-
      rich materials of ejecta deposits from the south pole-Aitken basin."
      Icarus, 358, 114214. https://doi.org/10/gmfc53
    """
    def r_flat(r_t, R, g, theta):
        """Return distance on flat surface, r' (eq S3, Zhang et al., 2021)"""
        v = ballistic_velocity(r_t * 1e3, cfg) / 1e3
        return R + (v**2 * np.sin(2 * np.deg2rad(theta))) / g

    def r_soi(r, R, R0):
        """Return distance to SOI (eq S4/S5, Zhang et al., 2021)."""
        tanr = np.tan((r - R) / (2 * R0))
        return R + (2 * R0 * tanr) / (tanr + 1)

    def s_flat(r_inner_sph, r_outer_sph):
        """Return area of flat SOI (eq S6, Zhang et al., 2021)."""
        return np.pi * (r_outer_sph**2 - r_inner_sph**2)

    def s_spherical(r_inner, r_outer, R0):
        """Return area of spherical SOI (eq S7, Zhang et al., 2021)."""
        return (
            2 * np.pi * R0**2 * (np.cos(r_inner / R0) - np.cos(r_outer / R0))
        )

    def area_corr(distance, radius, R0):
        """Return correction for area of SOI (eq S8, Zhang et al., 2021)."""
        r_inner = distance - radius / 8
        r_outer = distance + radius / 8
        area_sph = s_spherical(r_inner, r_outer, R0)
        r_inner_sph = r_soi(r_inner, radius, R0)
        r_outer_sph = r_soi(r_outer, radius, R0)
        area_flat = s_flat(r_inner_sph, r_outer_sph)
        return area_flat / area_sph

    # Compute dist and area corrections and return thickness
    exp = cfg.ej_thickness_exp
    radius = t_radius * 1e-3  # [m] -> [km]
    distance *= 1e-3  # [m] -> [km]
    R0 = cfg.rad_moon * 1e-3  # [m] -> [km]
    grav = cfg.grav_moon * 1e-3  # [m] -> [km]
    theta = cfg.impact_angle  # [deg]
    r_t = distance - radius
    rflat = r_flat(r_t, radius, grav, theta)
    with np.errstate(divide="ignore", over="ignore"):
        acorr = area_corr(distance, radius, R0)
        thickness = 0.033 * radius * np.power(rflat / radius, exp) * acorr
    return thickness * 1e3  # [km] -> [m]


def get_ejecta_thickness_time(time_arr, df, ej_distances, cfg):
    """Return ejecta thickness and ejecta sources at each time. 
    
    Ejecta layers smaller than cfg.thickness_min are set to 0.
    
    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    df : pandas.DataFrame
        Crater DataFrame.
    ej_distances : numpy.ndarray
        2D array of crater to cold trap distances [m].
    cfg : moonpies.config.Cfg, optional
        cfg.thickness_min : float
            Minimum ejecta layer thickness [m].

    Returns
    -------
    thickness_time : numpy.ndarray
        2D array ejecta thickness [m] at each time (Ntime, Ncoldtrap).
    sources_time : numpy.ndarray of str
        2D array of ejecta source(s) at each time (Ntime, Ncoldtrap).

    See Also
    --------
    get_ejecta_thickness_matrix
    """
    ej_thick = get_ejecta_thickness_matrix(df, ej_distances, cfg)
    ej_ages = df.age.values
    ej_thick_t = ages2time(time_arr, ej_ages, ej_thick, cfg, np.nansum, 0)

    # Label sources above threshold
    has_ej = ej_thick > cfg.thickness_min
    ej_srcs = (df.cname.values[:, np.newaxis] + ",") * has_ej
    ej_sources_t = ages2time(time_arr, ej_ages, ej_srcs, cfg, np.sum, "", object)
    ej_sources_t = np.char.rstrip(ej_sources_t.astype(str), ",")
    return ej_thick_t, ej_sources_t


def get_ejecta_thickness_matrix(df, ej_dists, cfg):
    """Return ejecta thickness from each crater to each cold trap.

    Rows are craters in df, columns are cold traps from cfg.coldtrap_names.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    ej_dists : numpy.ndarray
        2D array of crater to cold trap distances [m] (Ncrater, Ncoldtrap).

    Returns
    -------
    thickness : numpy.ndarray
        2D array of ejecta thickness [m] (Ncrater, Ncoldtrap).
    
    See Also
    --------
    get_ejecta_thickness_time, get_ejecta_thickness
    """
    # Get radius array same shape as ej_dists (radii as rows)
    rad = np.tile(df.rad.values[:, np.newaxis], (1, ej_dists.shape[1]))
    return get_ejecta_thickness(ej_dists, rad, cfg)


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
    # def insert_unique_in_range(src, target):
    #     """Return sorted target with unique values in src inserted."""
    #     uniq = np.unique(src[~np.isnan(src)])
    #     # Ensure in range of target
    #     uniq = uniq[(target.min() < uniq) & (uniq < target.max())]
    #     arr = np.sort(np.unique(np.concatenate([target, uniq])))
    #     return arr

    mdf = read_ballistic_melt_frac(cfg, True)
    
    # set up bivariate spline interpolation
    x = mdf.columns.to_numpy().astype(np.float64)
    y = mdf.index.to_numpy().astype(np.float64)
    nx = len(x)
    ny = len(y)
    xx, yy = np.meshgrid(x, y) # these have shape nx x ny
    xx = xx.reshape((nx*ny))
    yy = yy.reshape((nx*ny))
    zz = mdf.values.astype(np.float64).reshape((nx*ny))

    spl,_,_,_ = bisplrep(xx, yy, zz)

    melt_frac = np.zeros_like((mixing_ratios))
    n = mixing_ratios.shape[0]
    for i in range(len(ejecta_temps)):
        
        # get mixing ratios and ejecta temp for this crater
        curr_mix = copy.copy(mixing_ratios[i,...]).reshape((n*n))
        curr_temp = np.ones((n*n)) * ejecta_temps[i]

        # interpolate for these values
        melt_frac[i,...] = bisplev(curr_temp, curr_mix, spl).reshape((n,n))
    
    return melt_frac


    # temps = insert_unique_in_range(ejecta_temps, mdf.columns.to_numpy())
    # mrs = insert_unique_in_range(mixing_ratios, mdf.index.to_numpy())
    # mdf = mdf.reindex(index=mrs, columns=temps)
    # minterp = mdf.interpolate(axis=0).interpolate(axis=1)

    # # Interpolate melt_frac at each non-nan ejecta_temp, mixing_ratio
    # inds = np.argwhere(~np.isnan(mixing_ratios))
    # melt_frac = np.zeros_like(mixing_ratios)
    # for i, j in inds:
    #     melt_frac[i, j] = minterp.loc[mixing_ratios[i, j], ejecta_temps[i, j]]
    # return melt_frac


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
    ej_temp = ejecta_temp(df, cfg, mixing_ratio.shape)
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
