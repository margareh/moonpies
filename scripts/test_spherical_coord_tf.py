# date:     7-30-2025
# author:   margaret hansen
# purpose:  test spherical coordinate transform from center of moon to point on surface

import copy
import numpy as np
import pandas as pd

from make_psrs import spherical2cartesian, cartesian2spherical, latlon2enu

# KM to AU conversion
KM_AU = 149597870.700

# WKT string for lunar polar stereographic projection (from Haworth DEM file)
WKT_STR = """PROJCS["PolarStereographic Moon",GEOGCS["D_Moon",DATUM["D_Moon",SPHEROID["Moon_polarRadius",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]]"""


if __name__ == "__main__":

    # generate some data with points to test
    # these are points around the equator + poles + some others
    lats = np.array([0., 0., 0., 0., 90., -90., 45., 45., -45., -45., -90.])
    lons = np.array([0., 90., 180., 270., 0., 0., 0., 180., 90., 270., 180.])
    n = len(lats)
    # print(n) # 10

    # # load ephemeris data
    # cols = ['date', 'bl1' ,'bl2', 'obs_sublon', 'obs_sublat', 'sun_sublon', 'sun_sublat', 'sun_range', 'sun_rdot']
    # dtypes = {'date' : str,
    #           'bl1' : str,
    #           'bl2' : str,
    #           'obs_sublon' : np.float64, 
    #           'obs_sublat' : np.float64,
    #           'sun_sublon' : np.float64,
    #           'sun_sublat' : np.float64,
    #           'sun_range' : np.float64,
    #           'sun_rdot' : np.float64}
    # eph_df = pd.read_csv('/media/ssd/ThesisWork/Volatiles/JPL Horizons/sun_position_2004_2024_values_only.txt', names=cols, dtype=dtypes, parse_dates=['date'])
    # eph_df = eph_df[['date', 'sun_sublon', 'sun_sublat', 'sun_range']]
    # # print(len(eph_df)) # 7671

    # set ephemeris (sub-solar) points to the same as the other points for now
    eph = np.dstack((lons, lats)).reshape((n, 2))

    # everything at once!
    v_local = latlon2enu(eph[0,1], eph[0,0], 1e3 * KM_AU, lats, lons, deg=True)
    _, elev, azim = cartesian2spherical(v_local[0,...], v_local[1,...], v_local[2,...], deg=True)
    print(v_local.shape)
    # print(elev.shape) # 3 x 11 x 11 
    # print(azim.shape) # 
    print("Local ENU points (first ):")
    print(v_local.T)

    print("Local elevation and azimuth:")
    print(elev)
    print(azim)

    # set up arrays to store results in
    local_elev_azim = np.zeros((n, n, 2))

    # for ind, row in eph_df.iterrows():

    print("Previous version:")
    for i in range(1):

        # position of sun in cartesian coordinates centered on moon
        # elevation = latitude
        # azimuth = longitude
        x_moon, y_moon, z_moon = spherical2cartesian(1e3 * KM_AU, eph[i,1], eph[i,0], deg=True)

        # local grid point in cartesian coordinates centered on moon
        x_l, y_l, z_l = spherical2cartesian(1737400, lats, lons, deg=True)

        # convert lats and longs to radians
        lats_r = np.deg2rad(lats)
        lons_r = np.deg2rad(lons)

        # convert to local coordinates for each of the points in the grid 
        cos_th = np.cos(lats_r)
        sin_th = np.sin(lats_r)
        cos_phi = np.cos(lons_r)
        sin_phi = np.sin(lons_r)

        # below definition is based on uvw2enu from https://github.com/geospace-code/pymap3d/blob/main/src/pymap3d/ecef.py#L365
        R = np.tile(np.eye(3).reshape((3,3,1)), (1,1,n))
        R[0, 0, :] = -sin_phi
        R[0, 1, :] = cos_phi
        R[1, 0, :] = -sin_th * cos_phi
        R[1, 1, :] = -sin_th * sin_phi
        R[1, 2, :] = cos_th
        R[2, 0, :] = cos_th * cos_phi
        R[2, 1, :] = cos_th * sin_phi
        R[2, 2, :] = sin_th

        # # additional transform from ENU to ESD
        # R_esd = np.array([[1., 0., 0.],
        #                   [0., -1., 0.],
        #                   [0., 0., -1.]])
        # R_full = np.einsum('ij,jlk->ilk', R_esd, R)

        # transform point for the sun in cartesian coordinates from global to local ENU surface frame for each point in grid
        local_t = np.array([x_l, y_l, z_l]).reshape((3,n))
        diff = np.array([x_moon, y_moon, z_moon]).reshape((3,1)) - local_t
        v_local = np.einsum('ijk,jk->ik', R, diff) # n x 3 output
        print(v_local.T)
        # print(np.linalg.norm(v_local, axis=0)) # these are all > 0

        # apply rotation to normal vectors to see if they only have z components (how it should be)
        norms = local_t / np.linalg.norm(local_t, axis=0)
        norms_local = np.einsum('ijk,jk->ik', R, norms).T
        print("Local norms (should all have 1 z components):")
        print(norms_local)
        # all of these have 1 z components (norms pointing away from surface)

    # function version
    print("Function version:")
    v_local = latlon2enu(eph[0,1], eph[0,0], 1e3 * KM_AU, lats, lons, deg=True)
    print(v_local.T)

    # # convert back to spherical coordinates in local frame --> elevation and azimuth of sun in local frame
    _, elev, azim = cartesian2spherical(v_local[0,...], v_local[1,...], v_local[2,...], deg=True)
    print(elev)
    print(azim)

