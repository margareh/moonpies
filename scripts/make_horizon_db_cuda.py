#! /usr/bin/python3.9
# # date:     7-18-2025
# author:   margaret hansen
# purpose:  create horizon database for 1 mpp Haworth DEM

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from process_haworth_tif import load_tif
from raytrace.raytrace import raytrace_horizon


def make_horizon_db_cuda(dem, args):

    dem_data = dem[0]

    # Define desired azimuthal angles to use
    # azims = np.arange(0, 150, 1)
    azims = np.array([0])

    # Loop through azimuthal angles
    i = 0
    start = time.time()
    for a in azims:
        
        print("On azimuthal angle " + str(i+1) + " / " + str(azims.shape[0]) + " (" +str(a)+ ")")

        # Force a to be an array so we correctly convert it to a tensor in the raytrace_horizon call
        if not isinstance(a, np.ndarray):
            a = np.array([a])

        # Call to raytracing
        # dem_data = dem_data[6000:6100, 6000:6150]
        # print(dem_data.shape)
        # dem_data = dem_data[2000:8000, 2000:8000]
        elevs = raytrace_horizon(dem_data, a, res=args.res, max_range=args.max_range, min_elev=args.min_elev, elev_delta=args.elev_delta)
        elevs[np.abs(elevs-args.min_elev) < 0.0001] = np.nan
        print(elevs.shape) # expect to see (h,w)
        # print(np.min(elevs))
        # print(np.max(elevs))

        non_zeros = (np.abs(elevs) > 0.0001)
        print(np.sum(non_zeros))

        c_min = np.min(dem_data)
        c_max = np.max(dem_data)

        # plot results
        r = int(np.floor(args.max_range * 1000 / args.res))
        h,w = dem_data.shape
        # y_pts = np.array([r, r, h-r, h-r, r])
        # x_pts = np.array([r, w-r, w-r, r, r])
        dem_data_limit = dem_data[r:(h-r), r:(w-r)]
        plt.close()
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(dem_data_limit, cmap='terrain', vmin=c_min, vmax=c_max)
        # ax[0].plot(x_pts, y_pts, c='red')
        ax[1].imshow(elevs, cmap='Blues')
        plt.show()

        # Save the results
        np.savez_compressed(os.path.join(args.outpath, "horizon", "Haworth_1m_horizon_"+str(a[0])+".npz"), elevs=elevs)
        i += 1
        
    print("Elapsed time: %.2f" % (time.time() - start) + " s")


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/media/ssd/ThesisWork/Volatiles/SouthPoleData/Haworth_DEM_1mpp', help='Path to tif file')
    parser.add_argument('--outpath', type=str, default='/media/ssd/ThesisWork/Volatiles/processed', help='Path to save output to')
    parser.add_argument('--tiff', type=str, default='Lunar_LROnac_Haworth_sfs-dem_1m_v3.tif', help='Name of tif file to load')
    parser.add_argument('--eph', type=str, default='JPL Horizons/sun_position_2004_2024_values_only.txt', help='Path to ephemeris data file')
    parser.add_argument('--min_elev', type=float, default=-89, help='Min elevation in degrees for searching for horizon')
    parser.add_argument('--elev_delta', type=float, default=0.25, help='Elevation step size in degrees for horizon search')
    parser.add_argument('--max_range', type=float, default=4, help='Maximum range in km for horizon search')
    parser.add_argument('--res', type=float, default=1, help='Resolution of DEM')
    parser.add_argument('--plot', action='store_true', help='Flag to plot tif file')
    args = parser.parse_args()

    # if outpath doesn't exist, then make it
    if os.path.exists(args.outpath) == False:
        os.makedirs(args.outpath)

    if os.path.exists(os.path.join(args.outpath, 'horizon')) == False:
        os.makedirs(os.path.join(args.outpath, 'horizon'))
    
    # load DEM tiff file
    dem_data = load_tif(args)

    if args.plot:
        plt.imshow(dem_data)
        plt.show()

    # create horizon DB
    make_horizon_db_cuda(dem_data, args)

