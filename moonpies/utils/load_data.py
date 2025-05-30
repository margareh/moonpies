"""
Data loader functions for moonpies
"""

import os
import numpy as np
import pandas as pd
import rasterio as rs
from functools import lru_cache


def load_tifs(cfg, downsample=50):
    """Return numpy arrays with spatial datasets from cfg.psr_in and cfg.slope_in

    These tif files have a nominal resolution of 20 m per pixel
    """
    psr_data = rs.open(os.path.join(cfg.spatial_data_path, cfg.psr_in))
    psr = psr_data.read(1)

    slope_data = rs.open(os.path.join(cfg.spatial_data_path, cfg.slope_in))
    slope = slope_data.read(1)

    # if downsample > 1, then downsample the data
    if downsample > 1:
        new_n = int(psr.shape[0] / downsample)
        psr = psr.reshape((new_n, downsample, new_n, downsample)).sum(axis=1).sum(axis=2)
        slope = slope.reshape((new_n, downsample, new_n, downsample)).sum(axis=1).sum(axis=2)

    return psr, slope


def read_crater_list(cfg):
    """Return DataFrame of craters from cfg.crater_csv_in.

    Required columns, names and units in cfg.crater_csv_in:
        - 'lat': Latitude [deg]
        - 'lon': Longitude [deg]
        - 'diam': Diameter [km]
        - 'age': Crater age [Gyr]
        - 'age_low': Age error residual, lower (e.g., age - age_low) [Gyr]
        - 'age_upp': Age error residual, upper (e.g., age + age_upp) [Gyr]

    Optional columns in cfg.crater_csv_in:
        - 'psr_area': Permanently shadowed area of crater [km^2]

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        cfg.crater_csv_in : str
            Path to crater CSV file.
        cfg.crater_cols : tuple of str
            Tuple of column names in cfg.crater_csv_in.

    Returns
    -------
    pandas.DataFrame
        Crater DataFrame.
    """
    df = pd.read_csv(cfg.crater_csv_in, names=cfg.crater_cols, header=0)

    # Convert units, mandatory columns
    df["diam"] = df["diam"] * 1000  # [km -> m]
    df["rad"] = df["diam"] / 2
    df["age"] = df["age"] * 1e9  # [Gyr -> yr]
    df["age_low"] = df["age_low"] * 1e9  # [Gyr -> yr]
    df["age_upp"] = df["age_upp"] * 1e9  # [Gyr -> yr]

    # Define optional columns
    if "psr_area" in df.columns:
        df["psr_area"] = df.psr_area * 1e6  # [km^2 -> m^2]
    else:
        # Estimate psr area as 90% of crater area
        df["psr_area"] = 0.9 * np.pi * df.rad**2
    return df


def read_basin_list(cfg):
    """Return dataframe of craters from basin_csv path with columns names.

    Required columns, names and units in cfg.basin_csv_in:
        - 'lat': Latitude [deg]
        - 'lon': Longitude [deg]
        - 'diam': Diameter [km]
        - 'age': Crater age [Gyr]
        - 'age_low': Age error residual, lower (e.g., age - age_low) [Gyr]
        - 'age_upp': Age error residual, upper (e.g., age + age_upp) [Gyr]

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        cfg.basin_csv_in : str
            Path to basin CSV file.
        cfg.basin_cols : tuple of str
            Tuple of column names in cfg.basin_csv_in.

    Returns
    -------
    pandas.DataFrame
        Basin DataFrame.
    """
    df = pd.read_csv(cfg.basin_csv_in, names=cfg.basin_cols, header=0)

    # Convert units, mandatory columns
    df["diam"] = df["diam"] * 1000  # [km -> m]
    df["rad"] = df["diam"] / 2
    df["age"] = df["age"] * 1e9  # [Gyr -> yr]
    df["age_low"] = df["age_low"] * 1e9  # [Gyr -> yr]
    df["age_upp"] = df["age_upp"] * 1e9  # [Gyr -> yr]
    return df


def read_volcanic_species(cfg):
    """Return DataFrame of time, species mass from Table S3 [1]_.

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        cfg.nk_csv_in : str
            Path to Table S3 [1] CSV file.
        cfg.nk_cols : tuple of str
            Tuple of column names in cfg.nk_csv_in.
        cfg.species : str
            Species name (must be in nk_cols).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns "time" and species.
    
    References
    ----------

    .. [1] Needham, D. H., & Kring, D. A. (2017). "Lunar volcanism produced a 
       transient atmosphere around the ancient Moon." Earth and Planetary 
       Science Letters, 478, 175-178. 
       https://doi.org/10.1016/j.epsl.2017.09.002

    """
    df = pd.read_csv(cfg.nk_csv_in, names=cfg.nk_cols, header=4)
    df = df[["time", cfg.species]]
    df["time"] = df["time"] * 1e9  # [Gyr -> yr]
    df[cfg.species] = df[cfg.species] * 1e-3  # [g -> kg]
    df = df.sort_values("time", ascending=False).reset_index(drop=True)
    return df


def read_solar_luminosity(cfg):
    """Return DataFrame of time, solar luminosity from Table 2 [1]_.

    Parameters
    ----------
    cfg : moonpies.config.Cfg, optional
        cfg.bahcall_csv_in : str
            Path to CSV of Table 2 [1].

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns "time" and "luminosity".
    
    References
    ----------
    .. [1] Bahcall, J. N., Pinsonneault, M. H., & Basu, S. (2001). "Solar 
       Models: Current Epoch and Time Dependences, Neutrinos, and 
       Helioseismological Properties." The Astrophysical Journal, 555(2), 
       990-1012. https://doi.org/10/dv2q29

    """
    cols = ["time", "luminosity"]
    df = pd.read_csv(cfg.bahcall_csv_in, names=cols, header=1)
    df.loc[:, "time"] = (4.57 - df.loc[:, "time"]) * 1e9  # [Gyr -> yr]
    return df


@lru_cache(2)
def read_ballistic_melt_frac(cfg, mean=True):
    """Return DataFrame of ballistic sedimentation melt fractions.

    Parameters
    ----------
    mean : bool, optional
        If True, return mean melt fraction. If False, return standard dev
    cfg : moonpies.config.Cfg, optional
        cfg.bsed_frac_mean_in : str
            CSV of ballistic sedimentation melt fractions.

    Returns
    -------
    pandas.DataFrame
        Melt fractions (index: mixing ratio, cols: ejecta temperature).
    """
    csv = cfg.bsed_frac_mean_in if mean else cfg.bsed_frac_std_in
    df = pd.read_csv(csv, index_col=0, dtype=cfg.dtype)
    df.columns = df.columns.astype(cfg.dtype)
    return df


def read_ballistic_hop_csv(bhop_csv):
    """
    Return dict of ballistic hop efficiency of each coldtrap in bhop_csv.
    """
    return pd.read_csv(bhop_csv, header=0, index_col=0)