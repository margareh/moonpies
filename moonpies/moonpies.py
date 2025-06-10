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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from moonpies import config

from moonpies.utils.utils import vprint, clear_cache, get_coldtrap_dists, get_grid_arrays
from moonpies.utils.rv import get_rng, randomize_crater_ages, random_icy_basins
from moonpies.utils.load_data import read_crater_list, read_basin_list, load_tifs
from moonpies.utils.save_output import format_save_outputs, get_gc_dist_grid

from moonpies.processes.ballistic import get_bsed_depth, get_ejecta_thickness_time, get_ejecta_thickness
from moonpies.processes.impact import overturn_depth_time, get_ballistic_hop_coldtraps, get_impact_ice, get_impact_ice_comet, garden_ice_column, remove_ice_overturn
from moonpies.processes.volcanic import get_volcanic_ice
from moonpies.processes.solar_wind import get_solar_wind_ice


class MoonPIES():

    # initialize the simulation
    # this includes loading the data, reading the configuration (if provided),
    # and intiializing precomputed data structures and values
    def __init__(self, cfg=config.Cfg()):
        self.cfg = cfg

        # Setup phase
        vprint(cfg, "Initializing run...")
        clear_cache()
        rng = get_rng(cfg)

        # Setup time array
        n = int((cfg.timestart - cfg.timeend) / cfg.timestep)
        self.time_arr = np.linspace(cfg.timestart, cfg.timestep, n, dtype=cfg.dtype)

        # Setup ice distribution grid structure
        depthsize = int(cfg.depthmax / cfg.depthres)
        grdxsize_px = int(cfg.grdxsize / cfg.grdstep)
        grdysize_px = int(cfg.grdysize / cfg.grdstep)
        self.grdy, self.grdx = get_grid_arrays(cfg)
        self.ice_col_grid = np.zeros((self.grdy.shape[0], self.grdx.shape[1], depthsize))
        # print(self.ice_col_grid.shape) # 608 x 608 x 70

        # Setup crater list
        df_craters = read_crater_list(cfg)
        df_craters["isbasin"] = False
        df_craters["icy_impactor"] = "no"
        n_crater = len(df_craters)
        # print(n_crater) # 24

        df_basins = read_basin_list(cfg)
        df_basins["isbasin"] = True
        df_basins = random_icy_basins(df_basins, cfg, rng)
        n_basin = len(df_basins)
        # print(n_basin) # 27

        # Combine DataFrames and randomize ages
        df = pd.concat([df_craters, df_basins])
        self.df = randomize_crater_ages(df, cfg.timestep, rng)

        if not cfg.ejecta_basins:
            self.df[~self.df.isbasin].reset_index(drop=True)

        # Load the PSR and slope data
        # These are adjusted to have the same size and resolution as our grid
        self.psr, self.slope = load_tifs(cfg, cache=True)
        # print(self.psr.shape) # 608 x 608
        # print(self.slope.shape)

        # display the psr and slope data (to see what it looks like)
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(self.psr, cmap='binary')
        # ax[1].imshow(self.slope, cmap='coolwarm')
        # ax[0].axis('off')
        # ax[1].axis('off')
        # ax[0].set_title('PSRs')
        # ax[1].set_title('Slope')
        # plt.savefig('tif_files.png', bbox_inches='tight', dpi=100)
        # plt.close()

        # Pre-compute distances to each crater and masks for craters
        self.crater_dist_grid = get_gc_dist_grid(self.df, self.grdx, self.grdy, self.cfg, mask=False)
        crater_mask = np.zeros_like(self.crater_dist_grid)
        # print(crater_mask.shape) # 51 x 608 x 608

        cr_id = 0
        for i, row in self.df.iterrows():
            crater_mask[cr_id] = (self.crater_dist_grid[cr_id] <= row['rad'])
            cr_id += 1

        crater_mask_all = np.any(crater_mask[self.df['isbasin']==False], axis=0)
        basin_mask_all = np.any(crater_mask[self.df['isbasin']], axis=0)
        # print(crater_mask_all.shape) # 608 x 608

        # fig, ax = plt.subplots()
        # ax.imshow(self.psr, cmap='binary')
        # ax.imshow(crater_mask_all, cmap='Oranges', alpha=0.5)
        # ax.imshow(basin_mask_all, cmap='Blues', alpha=0.5)
        # ax.axis('off')
        # plt.savefig('craters_and_psrs.png', bbox_inches='tight', dpi=100)
        # plt.close()

        # Ejecta thickness produced by each crater on grid (3D array: NX, NY, NC)
        rad = self.df.rad.values[:, np.newaxis, np.newaxis]
        dists_masked = get_gc_dist_grid(self.df, self.grdx, self.grdy, self.cfg)
        self.ej_thick_grid = get_ejecta_thickness(dists_masked, rad, self.cfg)
        # print(self.ej_thick_grid.shape) # 51 x 608 x 608

        # TODO: review from here down to see how we can change the code to account for spatial distribution
        # Init strat columns dict based for all cfg.coldtrap_names
        self.ej_dists = get_coldtrap_dists(self.df, cfg)  # Crater -> coldtrap distances (2D)
        print(self.ej_dists.shape)
        
        # ej_cols, ej_srcs = get_ejecta_thickness_time(self.time_arr, self.df, self.ej_dists, self.cfg)
        
        # # Get column vectors of polar and volc ice
        # impact_ice = get_impact_ice(self.time_arr, self.df, cfg, rng)
        # comet_ice = get_impact_ice_comet(self.time_arr, self.df, cfg, rng)
        # if cfg.impact_ice_comets:
        #     # comet_ice is run every time for repro, but only add if needed
        #     impact_ice += comet_ice
        # solar_wind_ice = get_solar_wind_ice(self.time_arr, cfg)
        # ice_polar = (impact_ice + solar_wind_ice)[:, None]
        # ice_volcanic = get_volcanic_ice(self.time_arr, cfg)[:, None]

        # # one per time array
        # # print(impact_ice.shape) # 425
        # # print(solar_wind_ice.shape) # 425
        # # print(ice_volcanic.shape) # 425 x 1

        # if cfg.use_volc_dep_effcy:
        #     # Rescale by volc dep effcy, apply evenly to all coldtraps
        #     ice_volcanic *= cfg.volc_dep_effcy / cfg.ballistic_hop_effcy
        # else:
        #     # Treat as ballistically hopping polar ice
        #     ice_polar += ice_volcanic
        #     ice_volcanic *= 0

        # # Rescale by ballistic hop efficiency per coldtrap
        # if cfg.ballistic_hop_moores:
        #     bhops = get_ballistic_hop_coldtraps(list(cfg.coldtrap_names), cfg)
        #     bhops /= cfg.ballistic_hop_effcy
        #     ice_polar = bhops * ice_polar  # row * col -> 2D arr
        # else:
        #     ice_polar = np.tile(ice_polar, len(cfg.coldtrap_names))
        # ice_cols = ice_polar + ice_volcanic


        # # Get cold trap crater names (corresponds to columns in ej_cols, ice_cols)
        # self.ctraps = cfg.coldtrap_names

        # # Build strat columns as {cname: ice_col, ej_col, ej_src}
        # self.strat_cols = {
        #     coldtrap: [ice_cols[:, i], ej_cols[:, i], ej_srcs[:, i]]
        #     for i, coldtrap in enumerate(self.ctraps)
        # }
        
        # # Get gardening and bsed time arrays
        # self.bsed_depth, self.bsed_frac = get_bsed_depth(self.time_arr, self.df, self.ej_dists, cfg) # ballistic sedimentation depth, fraction by time
        # self.overturn = overturn_depth_time(self.time_arr, cfg) # overturn depth by time
        # # print(self.bsed_depth.shape) # 425 x 12 (time x coldtrap)
        # # print(self.bsed_frac.shape) # 425 x 12
        # # print(self.overturn.shape) # 425 (time)

        
    # update for one time step at a time
    def update(self, t, overturn_d):
        
        # Update all coldtrap ice_cols
        for i, coldtrap in enumerate(self.cfg.coldtrap_names):

            ice_col, ej_col, _ = self.strat_cols[coldtrap]

            # Ballistic sed gardens column before any ice gain (timestep t-1)
            ice_col = garden_ice_column(ice_col, ej_col, t - 1, self.bsed_depth[t,i], self.bsed_frac[t,i])

            # Ice "gained" by column (already pre-computed in ice_col[t])

            # Ice gardened at end of timestep, i.e. after ice gain (timestep t)
            ice_col = remove_ice_overturn(ice_col, ej_col, t, overturn_d, self.cfg)

            self.strat_cols[coldtrap][0] = ice_col  # Redundant (updated in place)


    # run through all time steps
    def run(self):
        vprint(self.cfg, "Starting main loop...")
        
        # Loop through all timesteps
        for t, overturn_t in enumerate(self.overturn):
            self.update(t, overturn_t)

    # save the output
    def save_output(self):
        return format_save_outputs(self.strat_cols, self.time_arr, self.df, self.cfg)

    # plot the output
    def show(self):
        pass


# main entrypoint function
def main(cfg):
    mp = MoonPIES(cfg)
    # mp.run()
    # return mp.save_output()


if __name__ == "__main__":
    cfg = config.Cfg()
    main(cfg)
