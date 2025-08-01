#! /usr/bin/python3.9
# date:     7-29-2025
# author:   margaret hansen
# purpose:  create psr map based on illumination data

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyproj import Proj, CRS

# KM to AU conversion
KM_AU = 149597870.700

# WKT string for lunar polar stereographic projection (from Haworth DEM file)
WKT_STR = """PROJCS["PolarStereographic Moon",GEOGCS["D_Moon",DATUM["D_Moon",SPHEROID["Moon_polarRadius",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]]"""


# conversion from spherical coordinates to cartesian
# r = range
# theta = elevation
# phi = azimuth
def spherical2cartesian(r, theta, phi, deg=False):
    if deg:
        # transform angles to radians
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

# conversion from spherical coordinates to cartesian
# x, y, z = cartesian coordinates
# theta is elevation angle
def cartesian2spherical(x, y, z, deg=False):

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arcsin(z / r)

    # define azimuth to be 0 if x and y are zero
    azim_0 = (np.abs(x) < 1e-15) * (np.abs(y) < 1e-15)
    # print(azim_0)
    phi = np.zeros_like(x)
    phi[~azim_0] = np.sign(y[~azim_0]) * np.arccos(x[~azim_0] / np.sqrt(x[~azim_0]**2 + y[~azim_0]**2))

    if deg:
        # return angles in degrees
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)

    return r, theta, phi


# convert from latitude and longitude to local ENU frame
def latlon2enu(lat, lon, range, lat0, lon0, deg=False, esd=False):

    # make sure all inputs are arrays
    if not isinstance(lat, np.ndarray):
        lat = np.array([lat])
    if not isinstance(lon, np.ndarray):
        lon = np.array([lon])
    if not isinstance(lat0, np.ndarray):
        lat0 = np.array(lat0)
    if not isinstance(lon0, np.ndarray):
        lon0 = np.array(lon0)

    # if deg = true, convert inputs to radians
    if deg:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        lat0 = np.deg2rad(lat0)
        lon0 = np.deg2rad(lon0)

    # position of sun in cartesian coordinates centered on moon
    # elevation = latitude
    # azimuth = longitude
    x_moon, y_moon, z_moon = spherical2cartesian(range, lat, lon)

    # local grid points in cartesian coordinates centered on moon
    x_l, y_l, z_l = spherical2cartesian(1737.4, lat0, lon0)

    # convert to local coordinates for each of the points in the grid 
    cos_th = np.cos(lat0)
    sin_th = np.sin(lat0)
    cos_phi = np.cos(lon0)
    sin_phi = np.sin(lon0)

    # below definition is based on uvw2enu from https://github.com/geospace-code/pymap3d/blob/main/src/pymap3d/ecef.py#L365
    n = len(lat0)
    R = np.tile(np.eye(3).reshape((3,3,1)), (1,1,n))
    R[0, 0, :] = -sin_phi
    R[0, 1, :] = cos_phi
    R[1, 0, :] = -sin_th * cos_phi
    R[1, 1, :] = -sin_th * sin_phi
    R[1, 2, :] = cos_th
    R[2, 0, :] = cos_th * cos_phi
    R[2, 1, :] = cos_th * sin_phi
    R[2, 2, :] = sin_th

    # additional transform from ENU to ESD if desired
    if esd:
        R_esd = np.array([[1., 0., 0.],
                        [0., -1., 0.],
                        [0., 0., -1.]])
        R = np.einsum('ij,jlk->ilk', R_esd, R)

    # transform point for the sun in cartesian coordinates from global to local ENU surface frame for each point in grid
    m = len(lat)
    local_t = np.array([x_l, y_l, z_l]).reshape((3,n))
    p = np.transpose(np.dstack((x_moon, y_moon, z_moon)), (0,2,1))
    a = np.tile(local_t.reshape((3,n,1)), (1,1,m))
    b = np.tile(p.reshape((3,1,m)), (1,n,1))
    diff = b - a
    R_all = np.tile(R.reshape((3,3,n,1)), (1,1,1,m))
    v_local = np.einsum('ijkl,jkl->ikl', R_all, diff)

    return np.squeeze(v_local)


# main function
def main(args):

    # load ephemeris data
    cols = ['date', 'bl1' ,'bl2', 'obs_sublon', 'obs_sublat', 'sun_sublon', 'sun_sublat', 'sun_range', 'sun_rdot']
    dtypes = {'date' : str,
              'bl1' : str,
              'bl2' : str,
              'obs_sublon' : np.float64, 
              'obs_sublat' : np.float64,
              'sun_sublon' : np.float64,
              'sun_sublat' : np.float64,
              'sun_range' : np.float64,
              'sun_rdot' : np.float64}
    eph_df = pd.read_csv(args.eph, names=cols, dtype=dtypes, parse_dates=['date'])
    eph_df = eph_df[['date', 'sun_sublon', 'sun_sublat', 'sun_range']]
    # print(len(eph_df)) # 7671
    
    # build a grid
    x = np.arange(0, args.size * args.res, args.res)
    x -= float(args.size) * args.res / 2 # center the grid values on 0
    XX, YY = np.meshgrid(x, x)
    # print(XX.shape) # 800 x 800
    grid = np.dstack((XX, YY)).reshape((args.size*args.size, 2))
    # print(grid.shape) # 640000 x 2

    # convert to lat long so we can compare to the ephemeris data
    # which projection is best? polar stereographic (ups) used for other data of south pole, gnomonic (gnom) would preserve geodesics as straight lines (do we care about this?)
    crs = CRS.from_wkt(WKT_STR)
    proj = Proj(crs)
    lons, lats = proj(grid[:,0], grid[:,1], inverse=True)

    for ind, row in eph_df.iterrows():

        # Compute position of sun relative to local ENU frames
        v_local = latlon2enu(row[['sun_sublat']], row[['sun_sublon']], row[['sun_range']], lats, lons, deg=True)
        _, elev, azim = cartesian2spherical(v_local[0,...], v_local[1,...], v_local[2,...], deg=True)

        # Now compare to horizon points for each elevation and azimuth



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--eph', type=str, default='/media/ssd/ThesisWork/Volatiles/JPL Horizons/sun_position_2004_2024_values_only.txt', help='Ephemeris file')
    parser.add_argument('--horizons', type=str, default='/media/ssd/ThesisWork/Volatiles/processed/synthterrain/horizon', help='Directory with horizons files')
    parser.add_argument('--res', type=float, default=1., help='Resolution of grid')
    parser.add_argument('--size', type=int, default=800, help='Size of grid (dimension of one side)')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to run for computing sun positions')
    args = parser.parse_args()

    # run the main function
    main(args)
