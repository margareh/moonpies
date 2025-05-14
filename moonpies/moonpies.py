"""Moon Polar Ice and Ejecta Stratigraphy model (MoonPIES).

This module contains the main functions for running the MoonPIES model. This
model simulates the evolution of polar ice and ejecta stratigraphy on the Moon
over geologic time.


:Authors: C.J. Tai Udovicic, K.R. Frizzell, K.M. Luchsinger, A. Madera, T.G. Paladino
:Acknowledgements: This model was developed as part of the 2021 Exploration
    Science Summer Intern Program hosted by the Lunar and Planetary Institute
    with support from the Center for Lunar Science and Exploration node of the 
    NASA Solar System Exploration Research Virtual Institute.

Edited by Margaret Hansen, 5-15-2025

"""

import os
import gc
from functools import lru_cache, _lru_cache_wrapper
import numpy as np
from scipy import stats
import pandas as pd
from moonpies import config


class MoonPIES():

    # initialize the simulation
    # this includes loading the data, reading the configuration (if provided),
    # and intiializing precomputed data structures and values
    def __init__(self, cfg=config.Cfg()):
        self.cfg = cfg

        # TODO: update code below and store important parts as class members
        # Setup phase
        vprint(cfg, "Initializing run...")
        clear_cache()
        rng = get_rng(cfg)

        # Setup time and crater list
        time_arr = get_time_array(cfg)
        df = get_crater_basin_list(cfg, rng)
        if not cfg.ejecta_basins:
            df[~df.isbasin].reset_index(drop=True)

        # Init strat columns dict based for all cfg.coldtrap_names
        ej_dists = get_coldtrap_dists(df, cfg)  # Crater -> coldtrap distances (2D)
        strat_cols = init_strat_columns(time_arr, df, ej_dists, cfg, rng)

        # Get gardening and bsed time arrays
        bsed_depth, bsed_frac = get_bsed_depth(time_arr, df, ej_dists, cfg)
        overturn = overturn_depth_time(time_arr, cfg)

        
    # update for one time step at a time
    def update(self):
        pass

    # run through all time steps
    def run(self):
        pass

    # save the output
    def save_output(self):
        pass

    # plot the output
    def show(self):
        pass


if __name__ == "__main__":

    mp = MoonPIES()
    mp.run()

