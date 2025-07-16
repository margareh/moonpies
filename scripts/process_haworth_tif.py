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

from moonpies.utils.utils import xy2latlon, latlon2xy
from urllib.request import urlopen

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


# query JPL Horizons to get range from surface point on moon to sun
def query_range_data(args):
    pass


# load the ephemeris data
def load_ephemeris_data(args):

    cols = ['timestamp', 'sp1', 'sp2', 'obs_lon', 'obs_lat', 'sun_lon', 'sun_lat']
    eph_df = pd.read_csv(os.path.join(args.path, args.eph), header=None, names=cols, index_col=False, parse_dates=['timestamp'])
    eph_df.drop(columns=['sp1','sp2'], inplace=True)
    # print(eph_df) # 7671 x 5
    return eph_df


# compute the apparent elevation and direction of the sun for a given lunar latitude and longitude
def compute_solar_elev(px_lat, px_lon, sun_lat, sun_lon, rp=1737.4e3, dp=1):

    # distance bewtween points in meters
    x_px, y_px = latlon2xy(px_lat, px_lon)
    x_sun, y_sun = latlon2xy(sun_lat, sun_lon)
    v_dir = np.array([x_sun - x_px, y_sun - y_px])
    d_c = np.sqrt(np.sum(v_dir**2)) # chordal distance

    # elevation = use law of cosines like 3 times
    cos_beta = 1 - (d_c**2) / (2*rp**2) # angle between points on lunar surface
    d = np.sqrt((rp**2) + (dp**2) - 2*rp*dp*cos_beta) # distance between sun and pixel 
    h = r - dp*cos_beta
    theta = np.arcsin(h / d) # elevation in radians

    return theta, v_dir


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
    for p in pixels:
        
        # query the JPL Horizons API for the range data for this point
        uri = "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='g:"+coords+"@301'&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='OBSERVER'&CENTER='500@10'&START_TIME='2004-01-01'&STOP_TIME='2024-12-31'&STEP_SIZE='1 DAYS'&REF_SYSTEM='ICRF'&CAL_FORMAT='CAL'&CAL_TYPE='G'&TIME_DIGITS='SECONDS'&CSV_FORMAT='YES'&QUANTITIES='19'"
        uri = uri.replace(' ', '%20').replace('&', '%24').replace(',', '%2C').replace(':', '%3A').replace('=','%3D').replace('?','%3F').replace('@','%40')
        page = urlopen(uri)
        text = page.read().decode("utf-8")
        print(text)


        # get sun elevation for all points in time relative to this pixel
        elev, dist = compute_solar_elev()

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

