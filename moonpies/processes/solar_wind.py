"""
Solar wind emplacement module
"""

import numpy as np
from moonpies.utils.load_data import read_solar_luminosity
from moonpies.utils.utils import get_ice_thickness


# Solar wind module
def get_solar_wind_ice(time_arr, cfg):
    """Return thickness of solar wind ice at all times.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        MoonPIES config object.

    Returns
    -------
    ice_time : numpy.ndarray
        Thickness of solar wind ice delivered [m] at each time.
    
    See Also
    --------
    solar_wind_ice_mass, get_ice_thickness
    """
    sw_ice_t = np.zeros_like(time_arr)
    if cfg.solar_wind_ice:
        sw_ice_mass = solar_wind_ice_mass(time_arr, cfg)
        sw_ice_t = get_ice_thickness(sw_ice_mass, cfg)
    return sw_ice_t


def solar_wind_ice_mass(time_arr, cfg):
    """Return global solar wind ice mass [kg] deposited at each time.

    H2O supply rate is given by cfg.solar_wind_mode ("Benna": 2g/s [1]_,
    "Lucey-Hurley": H2 supply of 30 g/s, H2 -> H2O at 1 ppt [2]_). If 
    cfg.faint_young_sun, scale solar wind H2O supply rate by relative solar
    luminosity at each time [3]_.

    Parameters
    ----------
    time_arr : numpy.ndarray
        Time array [yr].
    cfg : moonpies.config.Cfg, optional
        cfg.solar_wind_mode : str
            Solar wind ice mode {"Benna", "Lucey-Hurley"}.
        cfg.faint_young_sun : bool
            Use faint young sun model.
        
    Returns
    -------
    ice_mass : numpy.ndarray
        Global solar wind ice mass [kg] deposited at each time.

    References
    ----------
    .. [1] Benna, M., Hurley, D. M., Stubbs, T. J., Mahaffy, P. R., & Elphic,
       R. C. (2019). "Lunar soil hydration constrained by exospheric water
       liberated by meteoroid impacts." Nature Geoscience, 12(5), 333-338.
       https://doi.org/10/c4nz

    .. [2] Lucey, P. G., Costello, E. S., Hurley, D. M., Prem, P., Farrell,
       W. M., Petro, N., & Cable, M. L. (2020). Relative Magnitudes of Water
       Sources to the Lunar Poles. LPS LI, Abstract # 2319.
       Retrieved from https://www.hou.usra.edu/meetings/lpsc2020/pdf/2319.pdf

    .. [3] Bahcall, J. N., Pinsonneault, M. H., & Basu, S. (2001). "Solar 
       Models: Current Epoch and Time Dependences, Neutrinos, and 
       Helioseismological Properties." The Astrophysical Journal, 555(2), 
       990-1012. https://doi.org/10/dv2q29
    """ 
    if cfg.solar_wind_mode.lower() == "benna":
        # Benna et al. 2019; Arnold, 1979; Housley et al. 1973
        volatile_supply_rate = 2 * 1e-3  # [g/s -> kg/s]
    elif cfg.solar_wind_mode.lower() == "lucey-hurley":
        # Lucey et al. 2020, Hurley et al. 2017 (assume 1 ppt H2 -> H2O)
        volatile_supply_rate = 30 * 1e-3 * 1e-3  # [g/s - kg/s] * [1 ppt]
    else:
        msg = 'cfg.solar_wind_mode not one of {"Benna", "Lucey-Hurley"}'
        raise ValueError(msg)
    # convert to kg per timestep
    supply_rate_ts = volatile_supply_rate * 60 * 60 * 24 * 365 * cfg.timestep

    sw_ice_mass = np.ones_like(time_arr) * supply_rate_ts
    if cfg.faint_young_sun:
        # Import historical solar luminosity (Bahcall et al. 2001)
        df_lum = read_solar_luminosity(cfg)

        # Interpolate to time_arr, scale by solar luminosity at each time
        lum_time = np.interp(-time_arr, -df_lum.time, df_lum.luminosity)
        sw_ice_mass *= lum_time
    return sw_ice_mass
