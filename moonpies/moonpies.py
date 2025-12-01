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
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import Tensor
from torch import load as load_t
from gpytorch.likelihoods import GaussianLikelihood

from moonpies import config

from .utils.gp import GP, gp_predict
from .utils.utils import vprint, clear_cache, get_grid_arrays
from .utils.rv import get_rng, randomize_crater_ages, random_icy_basins
from .utils.load_data import read_crater_list, read_basin_list, load_tifs
from .utils.save_output import get_gc_dist_grid

from .processes.ballistic import get_ejecta_thickness, get_mixing_ratio_oberbeck, ejecta_temp
from .processes.impact import overturn_depth_time, get_impact_ice, get_impact_ice_comet
from .processes.volcanic import get_volcanic_ice
from .processes.solar_wind import get_solar_wind_ice


class MoonPIES():

    # initialize the simulation
    # this includes loading the data, reading the configuration (if provided),
    # and intiializing precomputed data structures and values
    def __init__(self, cfg=config.Cfg(), crater_db=None, basin_db=None, psr_mask=None):
        self.cfg = cfg

        # Setup phase
        vprint(cfg, "Initializing run...")
        clear_cache()
        self.rng = get_rng(cfg)

        # Setup time array
        self.time_arr = np.arange(cfg.timestart, cfg.timeend-cfg.timestep, -cfg.timestep, dtype=cfg.dtype)

        # Setup ice distribution grid structure
        # grdxsize_px = int(cfg.grdxsize / cfg.grdstep)
        # grdysize_px = int(cfg.grdysize / cfg.grdstep)
        self.grdy, self.grdx = get_grid_arrays(cfg, half=cfg.halfgrid)
        # this has a channel per time step (layer)
        self.ice_col_grid = np.zeros((len(self.time_arr), self.grdy.shape[0], self.grdx.shape[1]))
        self.ej_col_grid = np.zeros_like(self.ice_col_grid)
        # print(self.ice_col_grid.shape) # 608 x 608

        # Load data and initialize useful things
        self.psr_area = np.zeros((self.grdy.shape[0], self.grdx.shape[1]))
        self.update_crater_info(crater_db, basin_db, psr_mask)

        # Pre-compute overturn depth
        print("Getting overturn depth time...")
        self.overturn = overturn_depth_time(self.time_arr, self.cfg) # overturn depth by time
        # print(self.overturn.shape) # 425 (time)

        # initial values based on start time of sim
        self.t = float(self.cfg.timestart)
        self.t_ind = 0 # time index into time array

        # Compute initial ejecta thickness
        print("Getting initial values of ejecta and ice")
        # this is equivalent to the total ejecta thickness for all craters that have been formed
        # by a specified time t
        self.deliver_ejecta(init=True) # results stored in self.ej_col_grid

        # Compute initial amount of ice
        # TODO: compare to prior method of delivering ice
        # make sure that basin impacts are only being attributed to time steps that are close to the current one
        self.deliver_ice() # results stored in self.ice_col_grid

        # TODO: should we have an impact gardening step here?

        # Compute depth and fraction of ice
        self.get_depth_frac()

        # Need to start with t_ind = 1
        self.t_ind += 1

        # Load the data and GP for melt fraction interpolation
        print("Getting GP for melt fraction interpolation...")
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

        # save the initial results
        self.save_output()

    
    # update crater list and PSR masks
    def update_crater_info(self, crater_db=None, basin_db=None, psr_mask=None, new_ej=None, new_ice=None):

        # crater info
        if crater_db is None:
            df_craters = read_crater_list(self.cfg)
            # n_crater = len(df_craters)
            # print(n_crater) # 24
        else:
            df_craters = copy.copy(crater_db)
        
        df_craters["isbasin"] = False
        df_craters["icy_impactor"] = "no"

        if self.cfg.ejecta_basins or self.cfg.impact_ice_basins:
            if basin_db is None:
                df_basins = read_basin_list(self.cfg)
                # n_basin = len(df_basins)
                # print(n_basin) # 27
            else:
                df_basins = copy.copy(basin_db)

            df_basins["isbasin"] = True
            df_basins = random_icy_basins(df_basins, self.cfg, self.rng)

            # Combine DataFrames and randomize ages if upper and lower bounds are not the same
            # randomization function also sorts based on age and name
            df = pd.concat([df_craters, df_basins])
        else:
            df = copy.copy(df_craters)

        self.df = randomize_crater_ages(df, self.cfg.timestep, self.rng)
        # print(len(self.df))

        # if self.cfg.coldtrap_names is None:
        #     self.cfg.coldtrap_names = self.df.index

        if not self.cfg.ejecta_basins:
            self.df[~self.df.isbasin].reset_index(drop=True)

        # Load the PSR and slope data
        print("Loading PSR data...")
        if psr_mask is None:
            # These are adjusted to have the same size and resolution as our grid
            self.psr, self.slope = load_tifs(self.cfg, cache=True)
            # print(self.psr.shape) # 608 x 608
            # print(self.slope.shape)
        else:
            # currently slope isn't used, so we aren't losing out on that if it isn't provided
            self.psr = copy.copy(psr_mask)

        # TODO: add computation of in_crater and psr_area flags if not already in dataframe

        # Pre-compute distances to each crater and masks for craters
        print("Computing crater distances...")
        self.crater_dist_grid = get_gc_dist_grid(self.df, self.grdx, self.grdy, self.cfg, mask=False)
        self.crater_mask = np.zeros_like(self.crater_dist_grid)
        # print(self.grdx.shape)
        # print(self.crater_mask.shape) # 51 x 608 x 608

        print("Computing crater masks")
        self.coldtrap_flag = np.full((len(self.df)), False)
        cr_id = 0
        for i, row in self.df.iterrows():
            # self.crater_mask[cr_id,...] = (self.crater_dist_grid[cr_id] <= row['rad'])
            # print(np.array(row['in_crater']).shape)
            self.crater_mask[cr_id,...] = np.array(row['in_crater'])
            # if np.isin(row.cname, self.cfg.coldtrap_names):
            #     self.coldtrap_flag[cr_id] = True
            self.coldtrap_flag[cr_id] = (row['psr_area'] >= 0.0001)
            cr_id += 1

        self.coldtrap_inds = np.where(self.coldtrap_flag)[0]
        self.coldtrap_mask = self.crater_mask[self.coldtrap_inds,...] * np.expand_dims(self.psr == 1, axis=0)
        # print(self.coldtrap_mask.shape) # should be 12 x 608 x 608

        # Flag coldtraps and label coldtrap areas
        psr_area = np.zeros_like(self.coldtrap_mask)
        cr_id = ct_id = 0
        for i, row in self.df.iterrows():
            if self.coldtrap_flag[cr_id]:
                psr_area[ct_id] = row['psr_area'] * self.coldtrap_mask[ct_id]
                ct_id += 1
            cr_id += 1
        
        self.psr_area = np.maximum(np.max(psr_area, axis=0), self.psr_area)

        # fig, ax = plt.subplots(1,2, figsize=(20,10))
        # ax[0].imshow(np.any(self.crater_mask, axis=0), cmap='Oranges', alpha=0.5)
        # ax[0].imshow(np.any(self.coldtrap_mask, axis=0), cmap='Blues', alpha=0.5)
        # ax[1].imshow(self.psr_area)
        # plt.show()

        # # add in area that produces uniform distribution for remaining PSRs
        # psr_no_crater = (self.psr == 1) * ~np.any(self.coldtrap_mask, axis=0)
        # psr_px = np.sum(psr_no_crater)
        # self.psr_area[-1,psr_no_crater] = psr_px * (self.cfg.grdstep**2)

        # Ejecta thickness produced by each crater on grid (3D array: NX, NY, NC)
        print("Computing ejecta thicknesses...")
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

        # set up ballistic hop efficiency grid
        # first add the values for cold traps
        print("Getting ballistic hop efficiency grid...")
        self.bhops_grid = self.psr * self.cfg.ballistic_hop_effcy
        # bhops = get_ballistic_hop_coldtraps(list(self.cfg.coldtrap_names), self.cfg).reshape((n_ct, 1, 1))
        # bhops_ct = np.sum(bhops * self.coldtrap_mask, axis=0)

        # # now add values for remaining PSR regions based on constant value
        # psr_no_ct = self.psr & ~np.any(self.coldtrap_mask, axis=0)
        # self.bhops_grid = psr_no_ct * self.cfg.ballistic_hop_effcy + bhops_ct

        # Update previous ejecta and ice column values
        if new_ej is not None:
            self.ej_col_grid[:self.t_ind,...] = new_ej
        if new_ice is not None:
            self.ice_col_grid[:self.t_ind,...] = new_ice
        
        
    # update for one time step at a time
    def update(self, overturn_d):

        # Ejecta thickness updated
        vprint(self.cfg, "Deliver ejecta")
        self.deliver_ejecta()

        # Ballistic sed gardens column before any ice gain
        vprint(self.cfg, "Ballistic sedimentation")
        self.bsed_garden_ice(first_ejecta=False)

        # Ice "gained" by column
        # this updates self.ice_cols directly
        vprint(self.cfg, "Deliver ice")
        self.deliver_ice()

        # Ice gardened at end of timestep, i.e. after ice gain
        vprint(self.cfg, "Overturn ice")
        self.overturn_ice(overturn_d, first_ejecta=False)

        # Compute depth and fraction
        vprint(self.cfg, "Compute depth and fraction of ice")
        self.get_depth_frac()


    # run through all time steps
    def run(self):
        vprint(self.cfg, "Starting main loop...")
        
        # Loop through all timesteps
        i = 0
        while self.t > self.cfg.timeend:

            # decrement time step
            self.t -= self.cfg.timestep

            print("On time step %d" % (int(self.t)))
            self.update(self.overturn[i])
            
            # save output every nth timestep
            if i % self.cfg.save_every_n == 0:
                self.save_output()

            # increment counter and time index
            i += 1
            self.t_ind += 1

    # run through all time steps
    def run_between(self, start_time, end_time):
        vprint(self.cfg, "Starting main loop...")
        
        # Loop through all timesteps
        i = 0
        self.t = start_time
        while self.t > end_time:

            # decrement time step
            self.t -= self.cfg.timestep

            print("On time step %4.2f Myr" % (self.t / 1e6))
            self.update(self.overturn[i])
            
            # save output every nth timestep
            if i % self.cfg.save_every_n == 0:
                self.save_output()

            # increment counter and time index
            i += 1
            self.t_ind += 1

    # save the output
    def save_output(self):

        outpath = os.path.join(self.cfg.out_path, str(int(self.t)))
        
        if os.path.exists(outpath) == False:
            os.makedirs(outpath)
        
        self.show(out=outpath)

        # format_save_outputs(self.strat_cols, self.time_arr, self.df, self.cfg)
        # if self.t == self.cfg.timeend:
        #     # store everything
        #     np.savez(os.path.join(outpath, 'data.npz'),
        #             ice_depth=self.depth,
        #             ice_frac=self.frac,
        #             ice_col_grid=self.ice_col_grid,
        #             ej_col_grid=self.ej_col_grid,
        #             time_arr=self.time_arr)
        # else:
        #     # only store the depth
        #     np.savez(os.path.join(outpath, 'data.npz'),
        #             ice_depth=self.depth,
        #             ice_frac=self.frac,
        #             time_arr=self.time_arr)
        np.savez(os.path.join(outpath, 'data.npz'),
                 ice_depth=self.depth,
                 ice_frac=self.frac,
                 ice_col_grid=self.ice_col_grid,
                 ej_col_grid=self.ej_col_grid,
                 time_arr=self.time_arr)

    # plot some helpful things
    def show(self, out=None):
        
        if out is None:
            if os.path.exists(self.cfg.out_path) == False:
                os.makedirs(self.cfg.out_path)
            out = copy.copy(self.cfg.out_path)
        elif os.path.exists(out) == False:
            os.makedirs(out)

        # limit output to only craters within a specific time range
        craters = self.crater_mask[self.df['isbasin'] == False,...]
        crater_ages = self.df[self.df['isbasin'] == False]['age']
        basins = self.crater_mask[self.df['isbasin'],...]
        basin_ages = self.df[self.df['isbasin']]['age']
        crater_mask_all = np.any(craters[crater_ages >= int(self.t),...], axis=0)
        basin_mask_all = np.any(basins[basin_ages >= int(self.t),...], axis=0)

        # ice and ejecta
        ice_col_all = np.sum(self.ice_col_grid[:(self.t_ind+1),...], axis=0)
        ej_col_all = np.sum(self.ej_col_grid[:(self.t_ind+1),...], axis=0)

        # figure names
        # fig1_name = 'tif_files.png'
        fig2_name = 'craters_and_psrs.png'
        fig3_name = 'ice_and_ejecta.png'
        fig4_name = 'ice_depth_and_fraction.png'

        # lrbt
        # from tif file (pre-downsample): -304000, 304000, -304000, 304000
        # map_ext = [-304000, 304000, -304000, 304000]
        if self.cfg.halfgrid:
            map_ext = [-self.cfg.grdxsize, self.cfg.grdxsize, -self.cfg.grdysize, self.cfg.grdysize]
        else:
            map_ext = [0, self.cfg.grdxsize, 0, self.cfg.grdysize]

        # # display the psr and slope data (to see what it looks like)
        # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        # ax[0].imshow(self.psr, cmap='binary', extent=map_ext)
        # ax[1].imshow(self.slope, cmap='coolwarm', extent=map_ext)
        # # ax[0].axis('off')that cause ballisti
        # # ax[1].axis('off')
        # ax[0].set_title('PSRs')
        # ax[1].set_title('Slope')
        # plt.savefig(os.path.join(out, 'figs', fig1_name), bbox_inches='tight', dpi=100)
        # plt.close()

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.psr, cmap='binary', extent=map_ext)
        ax.imshow(crater_mask_all, cmap='Oranges', alpha=0.5, extent=map_ext)
        # ax.imshow(basin_mask_all, cmap='Blues', alpha=0.5, extent=map_ext)
        # ax.axis('off')
        plt.savefig(os.path.join(out, fig2_name), bbox_inches='tight', dpi=100)
        plt.close()

        # ice column
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
        im = ax[0].imshow(ice_col_all, cmap='Blues', extent=map_ext, vmin=0, vmax=self.cfg.vmax_ice)
        im2 = ax[1].imshow(ej_col_all, cmap='Oranges', extent=map_ext, vmin=0, vmax=self.cfg.vmax_ej)
        ax[0].set_title('Ice')
        ax[1].set_title('Ejecta')
        fig.colorbar(im, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])

        plt.savefig(os.path.join(out, fig3_name), dpi=100, bbox_inches='tight')
        plt.close()

        # ice depth and fraction
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            log_frac = np.log(self.frac)
        log_frac[np.isinf(log_frac)] = np.nan
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
        im = ax[0].imshow(log_frac, cmap='Blues', extent=map_ext, vmin=self.cfg.vmin_log_ice, vmax=0)
        im2 = ax[1].imshow(self.depth, cmap='Oranges', extent=map_ext, vmin=0, vmax=self.cfg.vmax_ej)
        ax[0].set_title('Log Ice Fraction')
        ax[1].set_title('Depth')
        fig.colorbar(im, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])

        plt.savefig(os.path.join(out, fig4_name), dpi=100, bbox_inches='tight')
        plt.close()


    # compute the ejecta thickness over spatial grid for a given time t
    def deliver_ejecta(self, init=False):
        
        # make sure time is same data type because otherwise functions below won't work
        t = np.array([self.t]).astype(self.cfg.dtype)
        ej_ages = self.df.age.values
        # print(ej_ages / 1e6)
        # print(t)
        # print(t+self.cfg.timestep)

        if init:
            crater_flag = (ej_ages >= t)
        else:
            # formed between previous time step and current one
            crater_flag = (ej_ages >= t) & (ej_ages < t+self.cfg.timestep)
        # print(crater_flag)

        ej_formed = self.ej_thick_grid[crater_flag, ...]
        # plt.imshow(np.sum(ej_formed, axis=0))
        # plt.show()
        self.ej_col_grid[self.t_ind,...] = np.sum(ej_formed, axis=0)


    # deliver ice for a given time step t
    def deliver_ice(self):

        # make sure time is same data type because otherwise functions below won't work
        t = np.array([self.t]).astype(self.cfg.dtype)

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

        # print(ice_volcanic.shape) # one value

        # Rescale by ballistic hop efficiency per coldtrap
        # TODO: should bhop efficiency be applied to full PSR? or should it be distributed 
        # throughout the PSR based on area or something like that so the total ice over the 
        # PSR represents the efficiency? (i.e. is it a ratio or is it an overall amount)
        # currently assuming we can use the efficiency at all locations in the PSR as is
        if self.cfg.ballistic_hop_moores:
            # need to remove the constant efficiency factor that was previously applied to ice
            ice_polar = self.bhops_grid * (ice_polar / self.cfg.ballistic_hop_effcy)
        else:
            ice_polar = np.ones_like(self.psr) * ice_polar

        # print(ice_polar.shape) # 608 x 608

        # filter to current set of psrs
        # TODO: is summing correct if there are overlapping sections? no --> less ice because area is in denominator
        # should instead apply ice from each PSR individually, so these regions would be more likely to end up with ice
        # PSR areas for craters + no-crater PSR areas
        no_psr = self.psr_area < 0.0001

        # adjust to be amount per pixel instead of total amount
        # want to set areas without specific ballistic hop efficiency to overall value
        ice_tot = ice_polar + ice_volcanic
        ice_tot[no_psr] = 0

        self.ice_col_grid[self.t_ind,...] = ice_tot
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
    def bsed_garden_ice(self, first_ejecta=False):
        
        # flag which craters were created during this time period
        t = np.array([self.t]).astype(self.cfg.dtype)
        ej_ages = self.df.age.values
        crater_flag = (ej_ages >= t) & (ej_ages < t+self.cfg.timestep)
        # print(self.df.cname.values)
        # print(self.df.lat.values)
        # print(self.df.lon.values)
        # print(crater_flag)
        
        # only run if we're using ballistic sedimentation and there are 
        # cratering events in this time period
        if len(np.argwhere(crater_flag)) > 0 and self.cfg.ballistic_sed:

            # compute ballistic sedimentation depth and fraction
            dists_t = self.dists_masked[crater_flag,...] # should be n x 608 x 608 with n = sum(crater_flag)
            mixing_ratio = get_mixing_ratio_oberbeck(dists_t, self.cfg) # n x 608 x 608
            # print(mixing_ratio.shape) # 1 x 608 x 608
            # print(np.min(mixing_ratio)) # 14.56 for first iter
            # print(np.max(mixing_ratio)) # 16.83 for first iter
            # curr_df = self.df[crater_flag]
            ej_temp = ejecta_temp(self.df[crater_flag], self.cfg) # n
            # print(ej_temp) # 1 value
            bsed_depths = self.ej_thick_grid[crater_flag,...] * mixing_ratio # Petro and Pieters (2004)
            # print(np.min(bsed_depths))
            # print(np.max(bsed_depths))
            # print(bsed_depths.shape) # n x 608 x 608
            p = bsed_depths.shape[-1] # 608

            # if there are nans for bsed_depth, replace with 0
            # these occur inside craters
            bsed_depths[np.isnan(bsed_depths)] = 0
            mixing_ratio[np.isnan(mixing_ratio)] = 0
            
            # interpolate melt fraction based on temperature and mixing ratio
            # these are sorted based on age descending so we can apply in order
            for i in range(np.sum(crater_flag)):
                curr_mix_r = mixing_ratio[i,...]
                curr_temp = ej_temp[i]
                inputs = np.dstack((np.ones_like(curr_mix_r) * curr_temp, curr_mix_r))
                # print(inputs.shape) # 608 x 608 x 2
                inputs = inputs.reshape((p*p,2))
                # print(inputs.shape)
                out_mean, _ = gp_predict(self.melt_frac_gp, self.lik, inputs, batch_size=1000, gpu=True)
                melt_frac = out_mean.reshape((p,p))
            
                # Scale by fraction lost from column (default 100%)
                melt_frac *= self.cfg.ballistic_sed_frac_lost

                # fig, ax = plt.subplots(2,2)
                # ax[0,0].imshow(dists_t[i,...])
                # ax[0,1].imshow(bsed_depths[i,...])
                # ax[1,0].imshow(mixing_ratio[i,...])
                # ax[1,1].imshow(melt_frac)
                # plt.show()

                # garden the ice column via ballistic sedimentation
                # this updates self.ice_col_grid in place
                # print(bsed_depths[i,...].shpae)
                # print(melt_frac.shape)
                self.garden_ice_d(bsed_depths[i,...], melt_frac, first_ejecta=first_ejecta, bsed=True)


    # gardening function applied to all ice column pixels based on provided depth and fraction
    # TODO: do we need to alternate ice and ejecta layers for this? can it be done simultaneously?
    def garden_ice_d(self, depth, eff=1, first_ejecta=False, bsed=False):

        # if only one depth value provided, use it everywhere
        if isinstance(depth, np.ndarray) == False:
            depth = np.ones_like(self.psr) * depth # 608 x 608

        # same with efficiency
        if isinstance(eff, np.ndarray) == False:
            eff = np.ones_like(self.psr) * eff # 608 x 608

        # remove ice with set efficiency to set depth
        # this assumes layers with ejecta over ice
        d = np.zeros_like(depth)
        needs_gardening = (d < depth)

        # for ballistic sedimentation, only want to look at previous timesteps, not current one
        if bsed:
            i = copy.copy(self.t_ind) - 1
        else:
            i = copy.copy(self.t_ind)

        while i >= 0 and np.any(needs_gardening):

            # if the first layer is ejecta, need to account for that
            if first_ejecta:
                # add ejecta to depth that's been gardened
                # also need to update flag for gardening here
                d += self.ej_col_grid[i,...]
                needs_gardening = (d < depth)

            # remove ice with amount capped after depth has been reached
            removed = self.ice_col_grid[i,...] * eff
            too_large = ((d + removed) > depth)
            removed[too_large] = depth[too_large] - d[too_large]

            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(removed)
            # ax[1].imshow(removed * needs_gardening)
            # plt.show()

            self.ice_col_grid[i,...] -= removed * needs_gardening

            # add remaining ice to depth
            d += self.ice_col_grid[i,...]

            # increment counter and recompute flag for needing gardening
            i -= 1
            needs_gardening = (d < depth)


    # alternate ice overturn function from Cannon
    # TODO: transfer and update function
    def erode_ice_cannon(self, erosion_depth=0.1, ej_shield=0.4):
        pass

    
    # overturn ice for a given time step t and overturn depth d
    def overturn_ice(self, d, first_ejecta=False):
        if self.cfg.impact_gardening_costello:
            self.garden_ice_d(d, first_ejecta=first_ejecta)
        else:
            self.erode_ice_cannon(erosion_depth=d)


# main entrypoint function
def main(cfg):
    mp = MoonPIES(cfg)
    mp.run()
    mp.save_output()


if __name__ == "__main__":
    cfg = config.Cfg()
    main(cfg)
