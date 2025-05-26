"""
Functions for saving output for moonpies
"""

import os
import numpy as np
import pandas as pd

from moonpies import config
from moonpies.utils.utils import vprint, xy2latlon, gc_dist, rtol, get_grid_arrays
from moonpies.processes.ballistic import get_ejecta_thickness


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
        unique_labels_l = list(unique_labels)
        unique_labels_l.sort()
        all_labels.append(",".join(unique_labels_l).strip(","))
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
