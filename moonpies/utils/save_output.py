"""
Functions for saving output for moonpies
"""

import os
import numpy as np
import pandas as pd

from moonpies import config
from moonpies.models.impactor import final2transient
from moonpies.processes.ballistic import ballistic_velocity
from moonpies.utils.utils import vprint, ages2time, xy2latlon, gc_dist, rtol, get_grid_arrays


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


def get_ice_columns_grid(strat_cols, df, grdx, grdy, cfg):
    """Return combined ice columns in grid form """
    dist_grid = get_gc_dist_grid(df, grdx, grdy, cfg, mask=False)
    
    # print(dist_grid.shape) # 51 x 801 x 801
    # print(strat_cols.keys()) # keys are the PSR names -> what about the non-named PSRs?
    # print(strat_cols['Haworth']) # 3 arrays: [ice, ej, ej_src]
    # print(strat_cols['Haworth'][0].shape) # 425 (this is time)

    # expand distance grid to be 

    return None

# Gridded outputs
def get_grid_outputs(strat_cols, df, grdx, grdy, cfg):
    """Return gridded age of most recent impact and ejecta thickness.

    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    grdx : numpy.ndarray
        Grid x-coordinates [m].
    grdy : numpy.ndarray
        Grid y-coordinates [m].
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.

    Returns
    -------
    age_grid : numpy.ndarray
        Gridded ages of last impact [yr] (Ny, Nx).
    thickness_grid : numpy.ndarray
        Gridded ejecta thickness [m] (Ny, Nx, Ncoldtraps).

    See Also
    --------
    get_age_grid, get_gc_dist_grid, get_ejecta_thickness_matrix
    """    
    # Age of most recent impact (2D array: NX, NY)
    age_grid = get_age_grid(df, grdx, grdy, cfg)

    # Great circle distance from each crater to grid (3D array: NX, NY, NC)
    dist_grid = get_gc_dist_grid(df, grdx, grdy, cfg)

    # Ejecta thickness produced by each crater on grid (3D array: NX, NY, NC)
    rad = df.rad.values[:, np.newaxis, np.newaxis]
    ej_thick_grid = get_ejecta_thickness(dist_grid, rad, cfg)

    # TODO: ballistic sed depth


    # TODO: kinetic energy

    # TODO: ice columns
    ice_col_grid = get_ice_columns_grid(strat_cols, df, grdx, grdy, cfg)

    return age_grid, ej_thick_grid, ice_col_grid


def get_gc_dist_grid(df, grdx, grdy, cfg, mask=True):
    """Return gridded great circle distance from each crater

    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    grdx : numpy.ndarray
        Grid x-coordinates [m].
    grdy : numpy.ndarray
        Grid y-coordinates [m].
    cfg : moonpies.config.Cfg, optional
        

    Returns
    -------
    distance_grid : numpy.ndarray
        Gridded from each crater in df [m] (Ny, Nx, Ncrater).

    See Also
    --------
    gc_dist
    """
    ny, nx = grdy.shape[0], grdx.shape[1]
    grdlat, grdlon = xy2latlon(grdx, grdy, cfg.rad_moon)
    grd_dist = np.zeros((len(df), nx, ny), dtype=cfg.dtype)
    for i, row in df.iterrows():
        clon, clat, crad = row[["lon", "lat", "rad"]]
        grd_dist[i] = gc_dist(clon, clat, grdlon, grdlat)

        if mask:
            cmask = grd_dist[i] < crad  # Mask crater interior
            thresh = cfg.basin_ej_threshold if row.isbasin else cfg.ej_threshold
            ejmask = grd_dist[i] > thresh * crad  # Mask ejecta exterior
            grd_dist[i] = np.where(cmask, np.nan, grd_dist[i])
            grd_dist[i] = np.where(ejmask, np.nan, grd_dist[i])

    return grd_dist


def get_age_grid(df, grdx, grdy, cfg):
    """Return final surface age of each grid point after all craters formed.

    Parameters
    ----------
    df : pandas.DataFrame
        Crater DataFrame.
    grdx : numpy.ndarray
        Grid x-coordinates [m].
    grdy : numpy.ndarray
        Grid y-coordinates [m].
    cfg : moonpies.config.Cfg, optional
        cfg.timestart : float
            Start time [yr].
        cfg.rad_moon : float
            Lunar radius [m].

    Returns
    -------
    age_grid : numpy.ndarray
        Gridded ages of last impact [yr] (Ny, Nx).
    """
    df = df.sort_values("age", ascending=False)  # sort oldest to youngest
    ny, nx = grdy.shape[0], grdx.shape[1]
    grdlat, grdlon = xy2latlon(grdx, grdy, cfg.rad_moon)
    age_grid = np.ones((nx, ny), dtype=cfg.dtype) * cfg.timestart
    for i, row in df.iterrows():
        lon, lat, rad, age, basin = row[["lon", "lat", "rad", "age", "isbasin"]]
        grd_dist = gc_dist(lon, lat, grdlon, grdlat)
        ej_thresh = cfg.basin_ej_threshold if basin else cfg.ej_threshold
        ejmask = grd_dist < rad * ej_thresh  # Mask ejecta blanket
        age_grid = np.where(ejmask, age, age_grid)  # Update age in ejecta
    return age_grid


# Format and export model results
def make_strat_col(time, ice, ejecta, ej_srcs, age, cfg):
    """
    Return stratigraphy column of ice and ejecta.

    Parameters
    ----------
    time (array): Time [yr]
    ice (array): Ice thickness [m]
    ejecta (array): Ejecta thickness [m]
    ej_srcs (array): Labels for ejecta layer sources
    age (num): Age at base of strat col [yr]

    Returns
    -------
    ej_col (DataFrame): Ejecta columns vs. time
    ice_col (array): Ice columns vs. time
    strat_cols (dict of DataFrame)
    """
    # Label rows by ice (no ejecta) or ejecta_source
    label = np.empty(len(time), dtype=object)
    label[ice > cfg.thickness_min] = "Ice"
    label[ejecta > cfg.thickness_min] = ej_srcs[ejecta > cfg.thickness_min]
    label[label == ""] = "Minor source(s)"

    # Make stratigraphy DataFrame
    data = np.array([time, ice, ejecta]).T
    cols = ["time", "ice", "ejecta"]
    sdf = pd.DataFrame(data, columns=cols, dtype=time.dtype)
    sdf["label"] = label

    # Exclude layers before age of crater
    if cfg.strat_after_age:
        sdf = sdf[sdf.time < age + rtol(age, cfg.rtol)].reset_index(drop=True)
        # Insert empty row labelling formation age of strat column
        sdf = pd.concat([
            pd.DataFrame([[sdf.iloc[0].time, 0, 0, "Formation age"]
            ], columns=sdf.columns), sdf], ignore_index=True)

    # Remove rows with no label (no ice or ejecta)
    sdf = sdf.dropna(subset=["label"])

    # Combine adjacent rows with same label into layers
    strat = merge_adjacent_strata(sdf)

    # Compute depth of each layer (reverse cumulative sum of ice and ejecta)
    strat["depth"] = np.cumsum((strat.ice + strat.ejecta)[::-1])[::-1]

    # Add ice / ejecta percentage of each layer
    strat["icepct"] = np.round(100 * strat.ice / (strat.ice + strat.ejecta), 4)
    return strat


def merge_adjacent_strata(strat, agg=None):
    """Return strat_col with adjacent rows with same label merged."""
    isadj = (strat.label != strat.label.shift()).cumsum()
    if agg is None:
        agg = {"ice": "sum", "ejecta": "sum", "label": "last", "time": "last"}
    return strat.groupby(["label", isadj], as_index=False, sort=False).agg(agg)


def format_csv_outputs(strat_cols, time_arr, df, cfg):
    """
    Return all formatted model outputs and write to outpath, if specified.

    Returns
    -------
    ejecta: pandas.DataFrame
        Ejecta thickness [m] (rows: time, cols: cold traps)
    ice: pandas.DataFrame
        Ice thickness [m] (rows: time, cols: cold traps)
    strat_dict: dict of str:pandas.DataFrame
        Dict of stratigraphy column DataFrames; keys: coldtrap names, values:
        dataframe (rows: layers, cols: ice, ejecta, source, time, depth, ice%)
    """
    ej_dict = {"time": time_arr}
    ice_dict = {"time": time_arr}
    strat_dfs = {}
    ej_source_all = []
    cdf = df.set_index('cname').loc[list(cfg.coldtrap_names), 'age']
    for cname, age in cdf.items():
        ice_t, ej_t, ej_srcs = strat_cols[cname]
        ej_dict[cname] = ej_t
        ice_dict[cname] = ice_t
        strat_dfs[cname] = make_strat_col(time_arr, ice_t, ej_t, ej_srcs, age, cfg)
        ej_source_all.append(ej_srcs)

    # Convert to DataFrames
    ej_cols_df = pd.DataFrame(ej_dict)
    ejecta_labels = get_all_labels(np.stack(ej_source_all, axis=1))
    ej_cols_df.insert(1, "ej_sources", ejecta_labels)
    ice_cols_df = pd.DataFrame(ice_dict)

    return ej_cols_df, ice_cols_df, strat_dfs


def get_all_labels(label_array):
    """Return all unique labels from label_array."""
    all_labels = []
    for label_col in label_array:
        all_labels_str = ",".join(label_col)
        unique_labels = set(all_labels_str.split(","))
        all_labels.append(",".join(unique_labels).strip(","))
    return all_labels


def format_save_outputs(strat_cols, time_arr, df, cfg):
    """
    Format dataframes and save outputs based on write / write_npy in cfg.
    """
    vprint(cfg, "Formatting outputs")
    ejdf, icedf, strat_dfs = format_csv_outputs(strat_cols, time_arr, df, cfg)
    if cfg.write:
        # Save config file
        vprint(cfg, f"Saving outputs to {cfg.out_path}")
        save_outputs([cfg], [cfg.config_py_out])

        # Save coldtrap strat column dataframes
        fnames = []
        dfs = []
        for coldtrap, strat in strat_dfs.items():
            fnames.append(os.path.join(cfg.out_path, f"strat_{coldtrap}.csv"))
            dfs.append(strat)
        save_outputs(dfs, fnames)

        # Save raw ice and ejecta column vs time dataframes
        save_outputs([ejdf, icedf], [cfg.ej_t_csv_out, cfg.ice_t_csv_out])
        print(f"Outputs saved to {cfg.out_path}")

    if cfg.write_npy:
        # Note: Only compute these on demand (expensive to do every run)
        # Age grid is age of most recent impact (2D array: NX, NY)
        vprint(cfg, "Computing gridded outputs...")
        grdy, grdx = get_grid_arrays(cfg)
        grd_outputs = get_grid_outputs(strat_cols, df, grdx, grdy, cfg)
        npy_fnames = (cfg.agegrd_npy_out, cfg.ejmatrix_npy_out, cfg.icecol_npy_out)
        vprint(cfg, f"Saving npy outputs to {cfg.out_path}")
        save_outputs(grd_outputs, npy_fnames)
    return strat_dfs


def save_outputs(outputs, fnames):
    """
    Save outputs to files in fnames in directory outpath.
    """
    for out, fout in zip(outputs, fnames):
        outpath = os.path.dirname(fout)
        if not os.path.exists(outpath):
            print(f"Creating new directory: {outpath}")
            os.makedirs(outpath)
        if isinstance(out, pd.DataFrame):
            out.to_csv(fout, index=False)
        elif isinstance(out, np.ndarray):
            np.save(fout, out)
        elif isinstance(out, config.Cfg):
            out.to_py(fout)
