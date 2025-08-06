"""
Random variables and RV mixture class for moonpies
"""

import numpy as np
from scipy import stats
from functools import lru_cache
from moonpies.utils.utils import round_to_ts

class rv_mixture(stats.rv_continuous):
    """Return mixture of scipy.stats.rv_continuous models."""

    def __init__(self, submodels, *args, weights=None, **kwargs):
        """
        Return mixture of scipy.stats rv submodels weighted by weights.

        Submodels must be instance of scipy.stats.rv_continuous. May not work
        for all stats.rv_continuous submodels. Tested on norm and truncnorm.

        Parameters
        ----------
        submodels (list of scipy.stats.rv_continuous): List of submodels
        weights (list of float): List of weights for submodels
        """
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise ValueError(
                f"Got {len(weights)} weights. Expected {len(submodels)}."
            )
        self.weights = [w / sum(weights) for w in weights]

    def _weighted_sum(self, vals):
        """Return weighted sum of vals (must be same len as self.weights)."""
        return np.sum(vals * np.array(self.weights, ndmin=2).T, axis=0)

    def _pdf(self, x, *args):
        pdfs = [submodel.pdf(x, *args) for submodel in self.submodels]
        return self._weighted_sum(np.array(pdfs))

    def _sf(self, x, *args):
        sfs = [submodel.sf(x, *args) for submodel in self.submodels]
        return self._weighted_sum(np.array(sfs))

    def _cdf(self, x, *args):
        cdfs = [submodel.cdf(x, *args) for submodel in self.submodels]
        return self._weighted_sum(np.array(cdfs))

    def _rvs(self, *args, size=None, random_state=None):
        size = 1 if size is None else size
        rng = np.random.default_rng() if random_state is None else random_state
        rv_choices = rng.choice(len(self.submodels), size=size, p=self.weights)
        rv_samples = [
            rv.rvs(*args, size=size, random_state=rng) for rv in self.submodels
        ]
        rvs = np.choose(rv_choices, rv_samples)
        return rvs

def _rng(rng):
    """Shorthand wrapper for np.random.default_rng."""
    return np.random.default_rng(rng)


def get_rng(cfg):
    """Return numpy random number generator from given seed in rng."""
    return _rng(cfg.seed)


def randomize_crater_ages(df, timestep, rng=None):
    """
    Return ages randomized with a truncated Gaussian between age_low, age_upp.

    Parameters
    ----------
    df (DataFrame): Crater dataframe with age, age_low, age_upp columns.
    timestep (int): Timestep [yr].
    rng (int or numpy.random.Generator): Seed or random number generator.
    """
    rng = _rng(rng)

    # If there's no difference between bounds, set age to the regular value
    diff = np.abs(df['age_upp'] - df['age_low'])

    # Set standard deviation and get scaling params for truncnorm
    sig = df[["age_low", "age_upp"]].mean(axis=1) / 2
    a = -df.age_low / sig
    b = df.age_upp / sig

    # Truncated normal, returns vector of randomized ages
    new_ages = stats.truncnorm.rvs(a, b, df.age, sig, random_state=rng)
    cond = np.abs(df.loc[:,'age_upp']-df.loc[:,'age_low']) > 0.0001
    df.loc[cond,"age"] = round_to_ts(new_ages[cond], timestep)
    df = df.sort_values(["age", "cname"], ascending=[False, True]).reset_index(drop=True)
    return df


def get_random_hydrated_craters(n, cfg, rng=None):
    """
    Return crater diams of hydrated craters from random distribution.

    Parameters
    ----------
    n (int): Number of craters.
    cfg (Cfg): Config object.
    rng (int or numpy.random.Generator): Seed or random number generator.
    """
    rng = _rng(rng)
    rand_arr = rng.random(size=n)
    if cfg.is_comet:
        # Get fraction of cometary impactors (assumed always hydrated)
        hydration_prob = cfg.comet_ast_frac
    elif cfg.impact_ice_comets:
        # Get fraction of asteroids, in runs that contain comets
        ast_frac = 1 - cfg.comet_ast_frac
        hydration_prob = ast_frac * cfg.ctype_frac * cfg.ctype_hydrated
    else:
        # Get fraction of ctype asteroids time prob of being hydrated
        hydration_prob = cfg.ctype_frac * cfg.ctype_hydrated
    hydrated_inds = rand_arr < hydration_prob
    return hydrated_inds


def get_random_hydrated_basins(n, cfg, rng=None):
    """
    Return indicess of hydrated basins from random distribution with asteroids
    and comets mutually exclusive.

    Parameters
    ----------
    n (int): Number of basins.
    cfg (Cfg): Config object.
    rng (int or numpy.random.Generator): Seed or random number generator.
    """
    rng = _rng(rng)
    rand_arr = rng.random(size=n)
    # Fraction of ctype hydrated basin impactors
    if cfg.impact_ice_comets:
        ast_frac = 1 - cfg.comet_ast_frac
        hydration_prob = ast_frac * cfg.ctype_frac * cfg.ctype_hydrated
    else:
        hydration_prob = cfg.ctype_frac * cfg.ctype_hydrated
    ctype_inds = rand_arr < hydration_prob

    # Fraction of cometary basins (use same rand_arr and pick unique inds)
    hydration_prob = cfg.comet_ast_frac
    comet_inds = rand_arr > (1 - hydration_prob)
    return ctype_inds, comet_inds


def random_icy_basins(df, cfg, rng=None):
    """Randomly assign each basin an impactor type determining hydration. 
    
    Probability of hydration and comet based on cfg. Possible icy_impactor 
    types are {"hyd_ctype", "comet", or "no"}.

    Parameters
    ----------
    df : pandas.DataFrame
        Basin DataFrame.
    cfg : moonpies.config.Cfg, optional
        cfg.impact_ice_basins : bool
            Whether to allow basins to be icy.
        cfg.impact_ice_comets : bool
            Whether to allow comet impactors.
    rng : int or numpy.random.Generator, optional
        Random seed or random number generator, by default None.

    Returns
    -------
    pandas.DataFrame
        Basin DataFrame with icy_impactor column added.

    See Also
    --------
    get_random_hydrated_basins
    """
    df["icy_impactor"] = "no"
    ctype, comet = get_random_hydrated_basins(len(df), cfg, rng)
    if cfg.impact_ice_basins:
        df.loc[ctype, "icy_impactor"] = "hyd_ctype"
        if cfg.impact_ice_comets:
            df.loc[comet, "icy_impactor"] = "comet"
    return df


def get_random_impactor_speeds(n, cfg, rng=None):
    """
    Return n impactor speeds from normal distribution about mean, sd.

    Parameters
    ----------
    n (int): Number of impactors.
    cfg (Cfg): Config object.
    rng (int or numpy.random.Generator): Seed or random number generator.
    """
    # Randomize impactor speeds with Gaussian around mean, sd
    rng = _rng(rng)
    rv = comet_speed_rv(cfg) if cfg.is_comet else asteroid_speed_rv(cfg)
    return rv.rvs(size=n, random_state=rng)


@lru_cache(maxsize=1)
def asteroid_speed_rv(cfg):
    """
    Return asteroid speed random variable object from scipy.stats.truncnorm.
    """
    # Set scaling param for stats.truncnorm (min impact speed is escape vel)
    a = (cfg.escape_vel - cfg.impact_speed_mean) / cfg.impact_speed_sd
    b = np.inf
    return stats.truncnorm(a, b, cfg.impact_speed_mean, cfg.impact_speed_sd)


@lru_cache(maxsize=1)
def comet_speed_rv(cfg):
    """
    Return comet speed random variable object from scipy.stats.truncnorm.
    """
    # Set scaling param for stats.truncnorm
    a = -cfg.comet_speed_min / cfg.jfc_speed_sd
    b = cfg.comet_speed_max / cfg.jfc_speed_sd
    jfc = stats.truncnorm(a, b, loc=cfg.jfc_speed_mean, scale=cfg.jfc_speed_sd)
    a = -cfg.comet_speed_min / cfg.lpc_speed_sd
    b = cfg.comet_speed_max / cfg.lpc_speed_sd
    lpc = stats.truncnorm(a, b, loc=cfg.lpc_speed_mean, scale=cfg.lpc_speed_sd)
    w = [cfg.jfc_frac, cfg.lpc_frac]
    return rv_mixture([jfc, lpc], weights=w)

