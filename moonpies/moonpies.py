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
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from moonpies import config

from moonpies.utils.utils import vprint, clear_cache, get_grid_arrays
from moonpies.utils.rv import get_rng, randomize_crater_ages, random_icy_basins
from moonpies.utils.load_data import read_crater_list, read_basin_list, load_tifs
from moonpies.utils.save_output import format_save_outputs, get_gc_dist_grid

from moonpies.processes.ballistic import get_ejecta_thickness, get_mixing_ratio_oberbeck, ejecta_temp, get_melt_frac
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
        self.rng = get_rng(cfg)

        # Setup time array
        n = int((cfg.timestart - cfg.timeend) / cfg.timestep)
        self.time_arr = np.linspace(cfg.timestart, cfg.timestep, n, dtype=cfg.dtype)
        # print(self.time_arr.shape) # 425

        # Setup ice distribution grid structure
        # grdxsize_px = int(cfg.grdxsize / cfg.grdstep)
        # grdysize_px = int(cfg.grdysize / cfg.grdstep)
        self.grdy, self.grdx = get_grid_arrays(cfg)
        # this has 2 channels for depth and ice pct
        self.ice_col_grid = np.zeros((self.grdy.shape[0], self.grdx.shape[1], 2))
        # print(self.ice_col_grid.shape) # 608 x 608

        # Setup crater list
        df_craters = read_crater_list(cfg)
        df_craters["isbasin"] = False
        df_craters["icy_impactor"] = "no"
        # n_crater = len(df_craters)
        # print(n_crater) # 24

        df_basins = read_basin_list(cfg)
        df_basins["isbasin"] = True
        df_basins = random_icy_basins(df_basins, cfg, self.rng)
        # n_basin = len(df_basins)
        # print(n_basin) # 27

        # Combine DataFrames and randomize ages
        # randomization function also sorts based on age and name
        df = pd.concat([df_craters, df_basins])
        self.df = randomize_crater_ages(df, cfg.timestep, self.rng)

        if not cfg.ejecta_basins:
            self.df[~self.df.isbasin].reset_index(drop=True)

        # Load the PSR and slope data
        # These are adjusted to have the same size and resolution as our grid
        self.psr, self.slope = load_tifs(cfg, cache=True)
        # print(self.psr.shape) # 608 x 608
        # print(self.slope.shape)

        # Pre-compute distances to each crater and masks for craters
        self.crater_dist_grid = get_gc_dist_grid(self.df, self.grdx, self.grdy, self.cfg, mask=False)
        self.crater_mask = np.zeros_like(self.crater_dist_grid)
        # print(self.crater_mask.shape) # 51 x 608 x 608

        cr_id = 0
        for i, row in self.df.iterrows():
            self.crater_mask[cr_id] = (self.crater_dist_grid[cr_id] <= row['rad'])
            cr_id += 1

        # Pre-compute distances to each coldtrap and masks for each coldtrap
        self.coldtrap_flag = np.zeros((len(self.df)))
        cr_id = 0
        for i, row in self.df.iterrows():
            if np.isin(row.cname, self.cfg.coldtrap_names):
                self.coldtrap_flag[cr_id] = 1
            cr_id += 1
        # print(self.coldtrap_flag.sum()) # 12
        # n_ct = len(self.cfg.coldtrap_names)

        self.coldtrap_inds = np.where(self.coldtrap_flag)[0]
        # print(self.df.iloc[self.coldtrap_inds])
        self.coldtrap_mask = self.crater_mask[self.coldtrap_inds,...] * np.expand_dims(self.psr, axis=0)
        # print(self.coldtrap_mask.shape) # should be 12 x 608 x 608

        # Ejecta thickness produced by each crater on grid (3D array: NX, NY, NC)
        rad = self.df.rad.values[:, np.newaxis, np.newaxis]
        # dists_masked = get_gc_dist_grid(self.df, self.grdx, self.grdy, self.cfg)
        self.dists_masked = copy.copy(self.crater_dist_grid)
        cr_id = 0
        for i, row in self.df.iterrows():
            mask = self.dists_masked[cr_id,...] < row[['rad']].values
            self.dists_masked[cr_id, mask] = np.nan
            cr_id += 1
        self.ej_thick_grid = get_ejecta_thickness(self.dists_masked, rad, self.cfg)
        # print(self.ej_thick_grid.shape) # 51 x 608 x 608

        # initial values based on start time of sim
        t_init = np.array([copy.copy(self.cfg.timestart)]).astype(self.cfg.dtype)

        # Compute initial ejecta thickness
        # this is equivalent to the total ejecta thickness for all craters that have been formed
        # by a specified time t
        self.get_ejecta_thickness_t(t_init) # results stored in self.ej_col

        # Compute initial amount of ice
        # TODO: compare to prior method of delivering ice
        # make sure that basin impacts are only being attributed to time steps that are close to the current one
        self.deliver_ice(t_init) # results stored in self.ice_cols

        # save in the ice grid!
        self.ice_col_grid[..., 0] = self.ice_cols + self.ej_col # depth
        self.ice_col_grid[..., 1] = self.ice_cols / self.ice_col_grid[..., 0] # ice fraction

        # # Pre-compute overturn depth
        self.overturn = overturn_depth_time(self.time_arr, self.cfg) # overturn depth by time
        # # print(self.overturn.shape) # 425 (time)

        
    # update for one time step at a time
    def update(self, t, overturn_d):
        
        # Ballistic sed gardens column before any ice gain
        # TODO: update using new bsed depth and fraction calcs
        self.garden_ice(t-self.cfg.timestep)

        # Ice "gained" by column
        # this updates self.ice_cols directly
        # self.deliver_ice(t)

        # Ice gardened at end of timestep, i.e. after ice gain
        # self.overturn_ice(t, overturn_d)

        # below is old code
        # for i, coldtrap in enumerate(self.cfg.coldtrap_names):

        #     self.ice_col, ej_col, _ = self.strat_cols[coldtrap]

        #     # Ballistic sed gardens column before any ice gain (timestep t-1)
        #     self.ice_col = garden_ice_column(self.ice_col, ej_col, t-cfg.timestep, self.bsed_depth[t,i], self.bsed_frac[t,i])

        #     # Ice "gained" by column
        #     self.deliver_ice(t)

        #     # Ice gardened at end of timestep, i.e. after ice gain (timestep t)
        #     self.ice_col = remove_ice_overturn(self.ice_col, ej_col, t, overturn_d, self.cfg)

        #     self.strat_cols[coldtrap][0] = self.ice_col  # Redundant (updated in place)


    # run through all time steps
    def run(self):
        vprint(self.cfg, "Starting main loop...")
        
        # Loop through all timesteps
        t = self.cfg.timestart + self.cfg.timestep # start with second timestep
        i = 0
        # while t < cfg.timeend:
        while t < self.cfg.timestart + 2*self.cfg.timestep:
            self.update(t, self.overturn[i])
            t += self.cfg.timestep
            i += 1
    

    # save the output
    def save_output(self):
        return format_save_outputs(self.strat_cols, self.time_arr, self.df, self.cfg)


    # plot some helpful things
    def show(self):
        
        if os.path.exists(self.cfg.out_path) == False:
            os.makedirs(self.cfg.out_path)

        # lrbt
        # from tif file (pre-downsample): -304000, 304000, -304000, 304000
        map_ext = [-304000, 304000, -304000, 304000]

        # display the psr and slope data (to see what it looks like)
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(self.psr, cmap='binary', extent=map_ext)
        ax[1].imshow(self.slope, cmap='coolwarm', extent=map_ext)
        # ax[0].axis('off')
        # ax[1].axis('off')
        ax[0].set_title('PSRs')
        ax[1].set_title('Slope')
        plt.savefig(os.path.join(self.cfg.out_path, 'tif_files.png'), bbox_inches='tight', dpi=100)
        plt.close()

        # crater mask over PSRs
        crater_mask_all = np.any(self.crater_mask[self.df['isbasin']==False], axis=0)
        basin_mask_all = np.any(self.crater_mask[self.df['isbasin']], axis=0)
        # print(crater_mask_all.shape) # 608 x 608

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.psr, cmap='binary', extent=map_ext)
        ax.imshow(crater_mask_all, cmap='Oranges', alpha=0.5, extent=map_ext)
        ax.imshow(basin_mask_all, cmap='Blues', alpha=0.5, extent=map_ext)
        # ax.axis('off')
        plt.savefig(os.path.join(self.cfg.out_path, 'craters_and_psrs.png'), bbox_inches='tight', dpi=100)
        plt.close()

        # ice column
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
        im = ax[0].imshow(self.ice_cols, cmap='Blues', extent=map_ext)
        im2 = ax[1].imshow(self.ej_col, cmap='Oranges', extent=map_ext)
        ax[0].set_title('Ice')
        ax[1].set_title('Ejecta')
        fig.colorbar(im, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])

        plt.savefig(os.path.join(self.cfg.out_path, 'ice_and_ejecta.png'), dpi=100, bbox_inches='tight')
        plt.close()

        # ice depth and fraction
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
        im = ax[0].imshow(self.ice_col_grid[...,0], cmap='Oranges', extent=map_ext)
        im2 = ax[1].imshow(self.ice_col_grid[...,1], cmap='Blues', extent=map_ext)
        ax[0].set_title('Depth')
        ax[1].set_title('Ice Fraction')
        fig.colorbar(im, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])

        plt.savefig(os.path.join(self.cfg.out_path, 'ice_depth_and_fraction.png'), dpi=100, bbox_inches='tight')
        plt.close()


    # compute the ejecta thickness over spatial grid for a given time t
    def get_ejecta_thickness_t(self, t):

        # ej_dists = distances between craters and cold traps
        # want to use actual distances between craters and grid locations
        # this should be stored in self.crater_dist_grid (note that interiors are not masked here)
        dists_masked = copy.copy(self.crater_dist_grid)
        cr_id = 0
        for i, row in self.df.iterrows():
            mask = dists_masked[cr_id,...] < row[['rad']].values
            dists_masked[cr_id, mask] = np.nan
            cr_id += 1
        
        ej_ages = self.df.age.values
        ej_formed = self.ej_thick_grid[(ej_ages <= t), ...]
        self.ej_col = np.sum(ej_formed, axis=0)


    # deliver ice for a given time step t
    def deliver_ice(self, t):

        impact_ice = get_impact_ice(t, self.df, self.cfg, self.rng)
        comet_ice = get_impact_ice_comet(t, self.df, self.cfg, self.rng)
        if self.cfg.impact_ice_comets:
            # comet_ice is run every time for repro, but only add if needed
            impact_ice += comet_ice
        solar_wind_ice = get_solar_wind_ice(t, self.cfg)
        ice_polar = (impact_ice + solar_wind_ice)[:, None]
        ice_volcanic = get_volcanic_ice(t, self.cfg)[:, None]
        # print(ice_polar.shape) # 1 x 1
        # print(ice_volcanic.shape) # 1 x 1

        if self.cfg.use_volc_dep_effcy:
            # Rescale by volc dep effcy, apply evenly to all coldtraps
            ice_volcanic *= self.cfg.volc_dep_effcy / self.cfg.ballistic_hop_effcy
        else:
            # Treat as ballistically hopping polar ice
            ice_polar += ice_volcanic
            ice_volcanic *= 0

        # Rescale by ballistic hop efficiency per coldtrap
        # TODO: should bhop efficiency be applied to full PSR? or should it be distributed 
        # throughout the PSR based on area or something like that so the total ice over the 
        # PSR represents the efficiency? (i.e. is it a ratio or is it an overall amount)
        # currently assuming we can use the efficiency at all locations in the PSR as is
        n_ct = len(self.cfg.coldtrap_names)
        if self.cfg.ballistic_hop_moores:
            bhops = get_ballistic_hop_coldtraps(list(self.cfg.coldtrap_names), self.cfg).reshape((n_ct,1,1))
            # print(bhops.shape) # 1 x 12
            bhops_grid = np.zeros_like(self.psr)
            bhops_grid = np.sum(bhops * self.coldtrap_mask, axis=0) # 608 x 608
            bhops_grid /= self.cfg.ballistic_hop_effcy
            ice_polar = bhops_grid * ice_polar
        else:
            ice_polar = np.ones_like(self.psr) * ice_polar
        
        # TODO: this is total per cold trap, currently assigning it to the same point in each CT
        # need to adjust so we aren't over-representing amount of ice
        self.ice_cols = ice_polar + ice_volcanic
        # print(self.ice_cols.shape) # 608 x 608


    # garden ice with ballistic sedimentation for a given time step t
    def garden_ice(self, t):
        
        # compute ballistic sedimentation depth and fraction
        # self.bsed_depth, self.bsed_frac = get_bsed_depth(t_init, self.df, self.ej_dists, cfg)
        if self.cfg.ballistic_sed:
        
            mixing_ratio = get_mixing_ratio_oberbeck(self.dists_masked, self.cfg) # 51 x 608 x 608
            ej_temp = ejecta_temp(self.df, self.cfg) # n_crater+n_basin
            melt_frac = get_melt_frac(ej_temp, mixing_ratio, self.cfg) # TODO: fix this
            print(melt_frac.shape) # 51 x 608 x 608
            bsed_depths = self.ej_thick_grid * mixing_ratio # Petro and Pieters (2004)
            print(bsed_depths.shape) # 51 x 608 x 608
            melt_frac *= self.cfg.ballistic_sed_frac_lost # Scale by fraction lost from column (default 100%)

        else:
            bsed_depths = np.zeros_like(self.psr)
            melt_frac = np.zeros_like(self.psr)

        # use in gardening the ice column
        # self.ice_col = garden_ice_column(self.ice_col, ej_col, t-1, self.bsed_depth[t,i], self.bsed_frac[t,i])
        
        pass

    
    # overturn ice for a given time step t and overturn depth d
    def overturn_ice(self, t, d):
        # self.ice_col = remove_ice_overturn(self.ice_col, ej_col, t, overturn_d, self.cfg)
        pass


# main entrypoint function
def main(cfg):
    mp = MoonPIES(cfg)
    mp.run()
    mp.show()
    # return mp.save_output()


if __name__ == "__main__":
    cfg = config.Cfg()
    main(cfg)
