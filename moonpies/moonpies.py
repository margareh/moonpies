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

from torch import Tensor
from torch import load as load_t
from gpytorch.likelihoods import GaussianLikelihood

from moonpies import config

from moonpies.utils.gp import GP, gp_predict
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
        # this has a channel per time step (layer)
        self.ice_col_grid = np.zeros((len(self.time_arr), self.grdy.shape[0], self.grdx.shape[1]))
        self.ej_col_grid = np.zeros_like(self.ice_col_grid)
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
        self.get_ejecta_thickness_t(t_init) # results stored in self.ej_col_grid

        # Compute initial amount of ice
        # TODO: compare to prior method of delivering ice
        # make sure that basin impacts are only being attributed to time steps that are close to the current one
        self.deliver_ice(t_init) # results stored in self.ice_col_grid

        # TODO: should we have an impact gardening step here?

        # Compute depth and fraction of ice
        self.get_depth_frac()

        # Pre-compute overturn depth
        self.overturn = overturn_depth_time(self.time_arr, self.cfg) # overturn depth by time
        # print(self.overturn.shape) # 425 (time)

        # Load the data and GP for melt fraction interpolation
        df = pd.read_csv(cfg.bsed_frac_mean_in, index_col=0, dtype=cfg.dtype)
        df.columns = df.columns.astype(cfg.dtype)
        x = df.columns.to_numpy()[1:].astype(np.float64)
        y = df.index.to_numpy()[1:].astype(np.float64)
        nx = len(x)
        ny = len(y)
        xx, yy = np.meshgrid(x, y) # these have shape nx x ny
        xx = xx.reshape((nx*ny))
        yy = yy.reshape((nx*ny))
        # skip the first row/column because they're all the same values
        zz = df.values[1:,1:].astype(np.float64).reshape((nx*ny))
        train_x = np.vstack((xx, yy)).T
        train_x_t = Tensor(train_x)
        train_z_t = Tensor(zz)

        gp_pth = cfg.bsed_frac_mean_in.replace('.csv', '_gp.pth')
        state_dict = load_t(gp_pth)
        self.lik = GaussianLikelihood()
        self.melt_frac_gp = GP(train_x_t, train_z_t, self.lik)
        self.melt_frac_gp.load_state_dict(state_dict)
        
        
    # update for one time step at a time
    def update(self, t, overturn_d):
        
        # Ejecta thickness updated
        vprint(self.cfg, "Deliver ejecta")
        self.get_ejecta_thickness_t(t)

        # Ballistic sed gardens column before any ice gain
        # TODO: update using new bsed depth and fraction calcs
        vprint(self.cfg, "Ballistic sedimentation")
        self.bsed_garden_ice(t)

        # Ice "gained" by column
        # this updates self.ice_cols directly
        vprint(self.cfg, "Deliver ice")
        self.deliver_ice(t)

        # Ice gardened at end of timestep, i.e. after ice gain
        vprint(self.cfg, "Overturn ice")
        self.overturn_ice(t, overturn_d)

        # Compute depth and fraction
        vprint(self.cfg, "Compute depth and fraction of ice")
        self.get_depth_frac()


    # run through all time steps
    def run(self):
        vprint(self.cfg, "Starting main loop...")
        
        # Loop through all timesteps
        t = self.cfg.timestart - self.cfg.timestep # start with second timestep
        i = 0
        while t > self.cfg.timeend:
            print("On time step %d" % (int(t)))
            self.update(t, self.overturn[i])
            t -= self.cfg.timestep
            i += 1
    

    # save the output
    def save_output(self, suffix=None):
        self.show(suffix=suffix)
        return format_save_outputs(self.strat_cols, self.time_arr, self.df, self.cfg)


    # plot some helpful things
    def show(self, suffix=None):
        
        if os.path.exists(self.cfg.out_path) == False:
            os.makedirs(self.cfg.out_path)

        # figure names
        if suffix is not None:
            fig1_name = 'tif_files_'+suffix+'.png'
            fig2_name = 'craters_and_psrs_'+suffix+'.png'
            fig3_name = 'ice_and_ejecta_'+suffix+'.png'
            fig4_name = 'ice_depth_and_fraction_'+suffix+'.png'
        else:
            fig1_name = 'tif_files.png'
            fig2_name = 'craters_and_psrs.png'
            fig3_name = 'ice_and_ejecta.png'
            fig4_name = 'ice_depth_and_fraction.png'

        # lrbt
        # from tif file (pre-downsample): -304000, 304000, -304000, 304000
        map_ext = [-304000, 304000, -304000, 304000]

        # display the psr and slope data (to see what it looks like)
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(self.psr, cmap='binary', extent=map_ext)
        ax[1].imshow(self.slope, cmap='coolwarm', extent=map_ext)
        # ax[0].axis('off')that cause ballisti
        # ax[1].axis('off')
        ax[0].set_title('PSRs')
        ax[1].set_title('Slope')
        plt.savefig(os.path.join(self.cfg.out_path, fig1_name), bbox_inches='tight', dpi=100)
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
        plt.savefig(os.path.join(self.cfg.out_path, fig2_name), bbox_inches='tight', dpi=100)
        plt.close()

        # ice column
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
        im = ax[0].imshow(np.sum(self.ice_col_grid, axis=0), cmap='Blues', extent=map_ext)
        im2 = ax[1].imshow(np.sum(self.ej_col_grid, axis=0), cmap='Oranges', extent=map_ext)
        ax[0].set_title('Ice')
        ax[1].set_title('Ejecta')
        fig.colorbar(im, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])

        plt.savefig(os.path.join(self.cfg.out_path, fig3_name), dpi=100, bbox_inches='tight')
        plt.close()

        # ice depth and fraction
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
        im = ax[0].imshow(self.depth, cmap='Oranges', extent=map_ext)
        im2 = ax[1].imshow(self.frac, cmap='Blues', extent=map_ext)
        ax[0].set_title('Depth')
        ax[1].set_title('Ice Fraction')
        fig.colorbar(im, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])

        plt.savefig(os.path.join(self.cfg.out_path, fig4_name), dpi=100, bbox_inches='tight')
        plt.close()


    # compute the ejecta thickness over spatial grid for a given time t
    def get_ejecta_thickness_t(self, t):
        
        # make sure time is same data type because otherwise functions below won't work
        if isinstance(t, np.ndarray) == False:
            t = np.array([t]).astype(self.cfg.dtype)
        ej_ages = self.df.age.values
        ej_formed = self.ej_thick_grid[(ej_ages <= t), ...]
        t_ind = np.argwhere(self.time_arr.astype(np.int64) == int(t))
        self.ej_col_grid[t_ind,...] = np.sum(ej_formed, axis=0)


    # deliver ice for a given time step t
    def deliver_ice(self, t):

        # make sure time is same data type because otherwise functions below won't work
        if isinstance(t, np.ndarray) == False:
            t = np.array([t]).astype(self.cfg.dtype)

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
        # need to adjust so we aren't over-representing amount of ice)
        t_ind = np.argwhere(self.time_arr.astype(np.int64) == int(t))
        t_ind = int(t_ind)
        self.ice_col_grid[t_ind,...] = ice_polar + ice_volcanic
        # print(self.ice_col_grid.shape) # 608 x 608


    # compute the depth of the first icy layer and the fraction of ice below that layer
    def get_depth_frac(self):

        no_ice_flag = (self.ice_col_grid < 0.0001)
        first_ice_ind = np.argmin(no_ice_flag, axis=0)
        first_ice_ind[np.sum(~no_ice_flag, axis=0) == 0,...] = self.ice_col_grid.shape[0]
        inds = np.indices(self.ice_col_grid.shape)[0,...]
        no_ice_flag[inds >= first_ice_ind] = False
        self.depth = np.sum(self.ej_col_grid * no_ice_flag, axis=0) # should be 608 x 608

        ice_tot = np.sum(self.ice_col_grid, axis=0)
        ej_tot = np.sum(self.ej_col_grid * ~no_ice_flag, axis=0) # only sum where there's ice
        self.frac = np.zeros_like(self.depth)
        # print(self.frac.shape)
        nonzero = (ice_tot + ej_tot > 0)
        # print(nonzero.shape)
        self.frac[nonzero] = ice_tot[nonzero] / (ice_tot[nonzero] + ej_tot[nonzero])
        self.frac[np.sum(~no_ice_flag, axis=0) == 0] = 0 # no ice if depth is max depth


    # garden ice with ballistic sedimentation for a given time step t
    def bsed_garden_ice(self, t):
        
        # flag which craters were created during this time period
        ej_ages = self.df.age.values
        crater_flag = (ej_ages > t-self.cfg.timestep) & (ej_ages <= t)
        
        # only run if we're using ballistic sedimentation and there are 
        # cratering events in this time period
        if len(np.argwhere(crater_flag)) > 0 and self.cfg.ballistic_sed:

            # compute ballistic sedimentation depth and fraction
            dists_t = self.dists_masked[crater_flag,...] # should be n x 608 x 608 with n = sum(crater_flag)
            mixing_ratio = get_mixing_ratio_oberbeck(dists_t, self.cfg) # n x 608 x 608
            curr_df = self.df[crater_flag]
            ej_temp = ejecta_temp(self.df[crater_flag], self.cfg) # n
            bsed_depths = self.ej_thick_grid[crater_flag,...] * mixing_ratio # Petro and Pieters (2004)
            # print(bsed_depths.shape) # n x 608 x 608
            p = bsed_depths.shape[-1] # 608
            
            # interpolate melt fraction based on temperature and mixing ratio
            # these are sorted based on age descending so we can apply in order
            for i in range(np.sum(crater_flag)):
                curr_mix_r = mixing_ratio[i,...]
                curr_temp = ej_temp[i]
                inputs = np.dstack((np.ones_like(curr_mix_r) * curr_temp, curr_mix_r)).reshape((p*p,2))
                # print(inputs.shape)
                out_mean, out_var = gp_predict(self.melt_frac_gp, self.lik, inputs, batch_size=1000, gpu=True)
                melt_frac = out_mean.reshape((p,p))
            
                # Scale by fraction lost from column (default 100%)
                melt_frac *= self.cfg.ballistic_sed_frac_lost

                # garden the ice column via ballistic sedimentation
                # this updates self.ice_col_grid in place
                self.garden_ice_d(t, bsed_depths[i,...], melt_frac)


    # gardening function applied to all ice column pixels based on provided depth and fraction
    # TODO: do we need to alternate ice and ejecta layers for this? can it be done simultaneously?
    def garden_ice_d(self, t, depth, eff=1):

        # make sure time is same data type because otherwise functions below won't work
        if isinstance(t, np.ndarray) == False:
            t = np.array([t]).astype(self.cfg.dtype)

        # if only one depth value provided, use it everywhere
        if isinstance(depth, np.ndarray) == False:
            depth = np.ones_like(self.psr) * depth # 608 x 608

        # same with efficiency
        if isinstance(eff, np.ndarray) == False:
            eff = np.ones_like(self.psr) * eff # 608 x 608
        
        # Travese ice and ejecta column from t down, removing ice, skipping ejecta
        # Loop until we hit the bottom or have gone down depth meters
        # - If ejecta[t] > depth, no ice is removed.
        # Double i so i//2 is current index to garden (odd: ejecta, even: ice)
        curr_time = np.argwhere(self.time_arr.astype(np.int64) == int(t))
        t_ind = int(curr_time)
        i = (2 * t_ind) + 1
        d = np.zeros_like(depth)  # current depth
        needs_gardening = (d < depth)
        while i >= 0 and np.any(needs_gardening):  # and < 2 * len(ice_column):
            if i % 2:
                # Odd i (ejecta): do nothing, add ejecta layer to depth, d
                d += self.ej_col_grid[i // 2,...]
            else:
                # Even i (ice): remove ice*eff from layer
                removed = self.ice_col_grid[i // 2, ...] * eff * needs_gardening

                # Removing more ice than depth, only remove enough to reach depth
                too_much = ((d + removed) > 0)
                removed[too_much] = depth[too_much] - d[too_much]
                self.ice_col_grid[i // 2, ...] -= removed
                d += self.ice_col_grid[i // 2,...]  # Count all ice in layer towards depth
            
            # increment counter and update flag for which pixels are done being gardened
            i -= 1
            needs_gardening = (d < depth)


    # alternate ice overturn function from Cannon
    # TODO: transfer and update function
    def erode_ice_cannon(self, t, erosion_depth=0.1, ej_shield=0.4):
        pass

    
    # overturn ice for a given time step t and overturn depth d
    def overturn_ice(self, t, d):
        if self.cfg.impact_gardening_costello:
            self.garden_ice_d(t, d)
        else:
            self.erode_ice_cannon(t, erosion_depth=d)


# main entrypoint function
def main(cfg):
    mp = MoonPIES(cfg)
    mp.run()
    mp.show()
    # return mp.save_output()


if __name__ == "__main__":
    cfg = config.Cfg()
    main(cfg)
