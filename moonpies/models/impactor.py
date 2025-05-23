"""
Impactor models for moonpies
"""

import numpy as np
from functools import lru_cache
from moonpies.utils.utils import diam2vol

# Crater-impactor scaling laws
def diam2len(
    diams,
    speeds,
    regime,
    cfg
):
    """
    Return size of impactors based on crater diams and impactor speeds.

    Different crater regimes are scaled via the following scaling laws:
    - regime=='c': (Prieur et al., 2017)
    - regime=='d': (Collins et al., 2005)
    - regime=='e': (Johnson et al., 2016)

    Parameters
    ----------
    diams (arr): Crater diameters [m].
    speeds (arr): Impactor speeds [m/s].
    regime (str): Crater scaling regime ('c', 'd', or 'e').

    Returns
    -------
    lengths (arr): Impactor diameters [m].
    """
    t_diams = final2transient(diams, cfg)
    if regime == "c":
        impactor_length = diam2len_prieur(
            tuple(t_diams),
            speeds,
            cfg.impactor_density,
            cfg.target_density,
            cfg.grav_moon,
            cfg.dtype,
        )
    elif regime == "d":
        impactor_length = diam2len_collins(
            t_diams,
            speeds,
            cfg.impactor_density,
            cfg.target_density,
            cfg.grav_moon,
            cfg.impact_angle,
        )
    elif regime in ("e", "f"):
        impactor_length = diam2len_johnson(
            t_diams,
            speeds,
            cfg.impactor_density,
            cfg.target_density,
            cfg.grav_moon,
            cfg.impact_angle,
            cfg.simple2complex,
        )
    # elif regime == "f": TODO: Implement Potter basin scaling
    else:
        raise ValueError(f"Invalid regime {regime} in diam2len")
    return impactor_length


def final2transient(diam, cfg):
    """
    Return transient crater diameters from final crater diams (Melosh 1989).

    Parameters
    ----------
    diams (num or array): final crater diameters [m]
    g (num): gravitational force of the target body [m s^-2]
    rho_t (num): target density (kg m^-3)

    Returns
    -------
    transient_diams (num or array): transient crater diameters [m]
    """
    # Init arrays and transition diams
    diam = np.atleast_1d(diam)
    t_diam = np.zeros_like(diam)
    ds2c = cfg.simple2complex
    dc2pr = cfg.complex2peakring

    # Simple craters
    mask = diam <= ds2c
    t_diam[mask] = f2t_melosh(diam[mask], simple=True)

    # Complex craters
    mask = (diam > ds2c) & (diam <= dc2pr)
    t_diam[mask] = f2t_melosh(diam[mask], simple=False)

    # Basins
    mask = diam > dc2pr
    t_diam[mask] = f2t_croft(diam[mask])  # TODO: make potter?
    return t_diam


def f2t_melosh(diam, simple=True, ds2c=18e3, gamma=1.25, eta=0.13):
    """
    Return transient crater diameter(s) from final crater diam (Melosh 1989).

    Parameters
    ----------
    diams (num or array): final crater diameters [m]
    simple (bool): if True, use simple scaling law (Melosh 1989).
    """
    if simple:  # Simple crater scaling
        t_diam = diam / gamma
    else:  # Complex crater scaling
        t_diam = (1 / gamma) * (diam * ds2c**eta) ** (1 / (1 + eta))
    return t_diam


def f2t_croft(diam, ds2c=18.7e3, eta=0.18):
    """
    Return transient crater diameter(s) fom final crater diam (Croft 1985).
    """
    return (diam * ds2c**eta) ** (1 / (1 + eta))


@lru_cache(1)
def diam2len_prieur(
    t_diam,
    v=20e3,
    rho_i=1300,
    rho_t=1500,
    g=1.62,
    dtype=None,
):
    """
    Return impactor length from input diam using Prieur et al. (2017) method.

    Note: Interpolates impactor lengths from the forward Prieur impactor length
    to transient crater diameter equation.

    Parameters
    ----------
    t_diam (num or array): transient crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)

    Returns
    -------
    impactor_length (num): impactor diameter [m]
    """
    i_lengths = np.linspace(t_diam[0] / 100, t_diam[-1], 1000, dtype=dtype)
    i_masses = rho_i * diam2vol(i_lengths)
    # Prieur impactor len to crater diam equation
    numer = 1.6 * (1.61 * g * i_lengths / v**2) ** -0.22
    denom = (rho_t / i_masses) ** (1 / 3)
    t_diams = numer / denom

    # Interpolate to back out impactor len from diam
    impactor_length = np.interp(t_diam, t_diams, i_lengths)
    return impactor_length


def diam2len_collins(
    t_diam,
    v=20e3,
    rho_i=1300,
    rho_t=1500,
    g=1.62,
    theta=45,
):
    """
    Return impactor length from input diam using Collins et al. (2005) method.

    Parameters
    ----------
    t_diam (num or array): transient crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)
    theta (num): impact angle (degrees)

    Returns
    -------
    impactor_length (num): impactor diameter [m]
    """
    cube_root_theta = np.sin(np.deg2rad(theta)) ** (1 / 3)
    denom = (
        1.161
        * (rho_i / rho_t) ** (1 / 3)
        * v**0.44
        * g**-0.22
        * cube_root_theta
    )
    impactor_length = (t_diam / denom) ** (1 / 0.78)
    return impactor_length


def diam2len_johnson(
    diam,
    v=20e3,
    rho_i=1300,
    rho_t=1500,
    g=1.62,
    theta=45,
    ds2c=18e3,
):
    """
    Return impactor length from final crater diam using Johnson et al. (2016).

    Parameters
    ----------
    diam (num or array): crater diameter [m]
    speeds (num): impact speed (m s^-1)
    rho_i (num): impactor density (kg m^-3)
    rho_t (num): target density (kg m^-3)
    g (num): gravitational force of the target body (m s^-2)
    theta (num): impact angle (degrees)

    Returns
    -------
    impactor_length (num): impactor diameter [m]
    """
    sin_theta = np.sin(np.deg2rad(theta))
    denom = (
        1.52
        * (rho_i / rho_t) ** 0.38
        * v**0.5
        * g**-0.25
        * ds2c**-0.13
        * sin_theta**0.38
    )
    impactor_length = (diam / denom) ** (1 / 0.88)
    return impactor_length
