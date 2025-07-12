# date:     7-11-2025
# author:   margaret hansen
# purpose:  process Haworth DTM to get illumination amounts


import os
import argparse
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rs

from moonpies.utils.utils import xy2latlon


# load and display tif file
def load_tif(args):

    # load the tif file
    data = rs.open(os.path.join(args.path, args.tiff))
    
    # print some metadata about the file
    # print(data.bounds) # geographic bounds (left-bottom-right-top)
    # print(data.width)
    # print(data.height)
    # print(data.count) # this is the number of bands
    # print(data.crs) # projection EPSG code
    # print(data.nodata) # value representing no data
    # 269XX = UTM north zone XX, NAD83
    # 326XX = UTM north zone XX, WGS 84
    # 4326 = LLA, WGS 84

    meta = {'bounds'    : data.bounds,
            'width'     : data.width,
            'height'    : data.height,
            'count'     : data.count,
            'crs'       : data.crs,
            'transform' : data.transform}

    # read the data and mask out missing values
    data_mask = data.read(1)
    data_mask[data_mask == data.nodata] = np.nan

    # either save or show a picture of it
    head, tail = os.path.split(args.tiff)
    fname = tail.replace('.tif', '.png')
    ext = [data.bounds[0], data.bounds[2], data.bounds[1], data.bounds[3]] # lrbt
    plt.imshow(data_mask, cmap='gray', extent=ext)
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.outpath, fname), dpi=100, bbox_inches='tight')
        plt.close()

    # close the dataset
    data.close()

    # return the loaded data and associated metadata
    return data_mask, meta


# load the ephemeris data
def load_ephemeris_data(args):

    cols = ['timestamp', 'sp1', 'sp2', 'obs_lon', 'obs_lat', 'sun_lon', 'sun_lat']
    # types = {'timestamp' : np.datetime64,
    #          'sp1' : str,
    #          'sp2' : str,
    #          'obs_lon' : np.float32,
    #          'obs_lat' : np.float32,
    #          'sun_lon' : np.float32,
    #          'sun_lat' : np.float32}
    eph_df = pd.read_csv(os.path.join(args.path, args.eph), header=None, names=cols, index_col=False, parse_dates=['timestamp'])
    eph_df.drop(columns=['sp1','sp2'], inplace=True)
    # print(eph_df) # 7671 x 5
    return eph_df


# compute the apparent elevation and direction of the sun for a given lunar latitude and longitude
def compute_solar_elev(px_lat, px_lon, sun_lat, sun_lon):
    pass


# process tif file to produce illumination values
def process_data(data, meta, eph, args):

    # get lat lon coordinates for metric units covered by dataset
    h = meta['height'] # y
    w = meta['width'] # x
    bounds = meta['bounds']
    # print(meta['transform'])
    xres = np.abs(meta['transform'][0])
    yres = np.abs(meta['transform'][4])
    # print(xres)
    # print(yres)
    # these are centers of the pixels in meters
    x = np.arange(bounds[0]+0.5*xres, bounds[2], xres)
    y = np.arange(bounds[1]+0.5*yres, bounds[3], yres)
    XX, YY = np.meshgrid(x, y)
    print(XX.shape) # 12060 x 11660

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(XX)
    ax[0].set_title('X')
    ax[1].imshow(YY)
    ax[1].set_title('Y')
    plt.show()

    coords_m = np.dstack((XX, YY)).reshape(h*w, 2)
    lat, lon = xy2latlon(coords_m[:,0], coords_m[:,1])
    coords = np.vstack((lat, lon)).reshape(h, w, 2)
    
    illumin_pct = np.zeros((pixels.shape[0]))
    # compute illumination conditions for each point
    # for p in pixels:
        
        # get sun elevation for all points in time relative to this pixel

        # compute local horizon in direction of sun

        # compare and update illumination percentage for this pixel

      

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/media/ssd/ThesisWork/Volatiles', help='Path to tif file')
    parser.add_argument('--outpath', type=str, default='/media/ssd/ThesisWork/Volatiles/processed', help='Path to save output to')
    parser.add_argument('--tiff', type=str, default='SouthPoleData/Haworth_DEM_1mpp/Lunar_LROnac_Haworth_sfs-dem_1m_v3.tif', help='Name of tif file to load')
    parser.add_argument('--eph', type=str, default='JPL Horizons/sun_position_2004_2024_values_only.txt', help='Path to ephemeris data file')
    parser.add_argument('--plot', action='store_true', help='Flag to plot tif file')
    args = parser.parse_args()

    # if outpath doesn't exist, then make it
    if os.path.exists(args.outpath) == False:
        os.makedirs(args.outpath)

    # load the tif data
    data, meta = load_tif(args)
    # this data is 11660 x 12060 (w x h)
    # each pixel is 1 m x 1 m resolution
    # polar stereographic projection

    # load the ephemeris data
    eph = load_ephemeris_data(args)

    # process the data
    process_data(data, meta, eph, args)

