#! /usr/bin/python3.9
# date:     7-29-2025
# author:   margaret hansen
# purpose:  create psr map based on illumination data

import os
import argparse
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyproj import Proj, CRS

# KM to AU conversion
KM_AU = 149597870.700

# WKT string for lunar polar stereographic projection (from Haworth DEM file)
WKT_STR = """PROJCS["PolarStereographic Moon",GEOGCS["D_Moon",DATUM["D_Moon",SPHEROID["Moon_polarRadius",1737400,0]],PRIMEM["Reference_Meridian",180],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",NORTH],AXIS["Northing",NORTH]]"""


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
    # extra pi added here to make azimuth match horizon database azimuth values
    azim_0 = (np.abs(x) < 1e-15) * (np.abs(y) < 1e-15)
    # print(azim_0)
    phi = np.zeros_like(x)
    phi[~azim_0] = np.sign(y[~azim_0]) * np.arccos(x[~azim_0] / np.sqrt(x[~azim_0]**2 + y[~azim_0]**2)) + np.pi

    if deg:
        # return angles in degrees
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)

    return r, theta, phi


# convert from latitude and longitude to local ENU frame
# lat, lon are the subsolar points
# range is the range of the sun
# lat0, lon0 are the local grid point(s)
def latlon2enu(lat, lon, range, lat0, lon0, deg=False, esu=False):

    # make sure all inputs are arrays
    if not isinstance(lat, np.ndarray):
        lat = np.array([lat])
    if not isinstance(lon, np.ndarray):
        lon = np.array([lon])
    if not isinstance(lat0, np.ndarray):
        lat0 = np.array([lat0])
    if not isinstance(lon0, np.ndarray):
        lon0 = np.array([lon0])

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
    x_l, y_l, z_l = spherical2cartesian(1737400, lat0, lon0)

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

    # transform point for the sun in cartesian coordinates from global to local ENU surface frame for each point in grid
    m = len(lat)
    local_t = np.array([x_l, y_l, z_l]).reshape((3,n))
    p = np.transpose(np.dstack((x_moon, y_moon, z_moon)), (0,2,1))
    a = np.tile(local_t.reshape((3,n,1)), (1,1,m))
    b = np.tile(p.reshape((3,1,m)), (1,n,1))
    diff = b - a
    R_all = np.tile(R.reshape((3,3,n,1)), (1,1,1,m))
    v_local = np.einsum('ijkl,jkl->ikl', R_all, diff)

    # additional transform from ENU to ESU (left-handed frame with x east, y down) if desired
    if esu:
        v_local[1,...] *= -1

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
    eph_df = pd.read_csv(args.eph, index_col=False, names=cols, dtype=dtypes, parse_dates=['date'], sep=',')
    eph_df = eph_df[['date', 'sun_sublon', 'sun_sublat', 'sun_range']]
    # print(len(eph_df)) # 7671
    
    # build a grid
    x = np.arange(0, args.size * args.res, args.res)
    x -= float(args.size) * args.res / 2 # center the grid values on 0
    XX, YY = np.meshgrid(x, -x)
    # print(XX.shape) # 800 x 800
    grid = np.dstack((XX, YY)).reshape((args.size*args.size, 2))
    # print(grid.shape) # 640000 x 2

    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(XX)
    # ax[1].imshow(YY)
    # plt.show()

    # convert to lat long so we can compare to the ephemeris data
    # which projection is best? polar stereographic (ups) used for other data of south pole, gnomonic (gnom) would preserve geodesics as straight lines (do we care about this?)
    crs = CRS.from_wkt(WKT_STR)
    proj = Proj(crs)
    lons, lats = proj(-grid[:,1], grid[:,0], inverse=True)
    lons += 180

    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(lons.reshape((args.size, args.size)))
    # ax[1].imshow(lats.reshape((args.size, args.size)))
    # plt.show()

    # load the horizons database
    horizon_db = np.zeros((args.size, args.size, 360))
    for i in range(360):
        elevs = np.squeeze(np.load(os.path.join(args.horizons, 'horizon_' + str(i) + '.npz'))['elevs'])
        horizon_db[...,i] = elevs
    del(elevs)

    # Initialize output array and calculate some useful things
    sum_illumin = np.zeros((args.size, args.size))
    sun_rad_deg = args.solar_disc_angle / 2
    sun_rad_sq = sun_rad_deg ** 2
    sun_area_degsq = np.pi * sun_rad_sq

    # loop through ephemeris and calculate elevation and azimuth for south pole
    # so we can graph it over time
    elevs = []
    azims = []
    for _, row in eph_df.iterrows():
        
        v_local = latlon2enu(row['sun_sublat'], row['sun_sublon'], row['sun_range'] * KM_AU * 1000, -89., 1., deg=True)
        _, elev, azim = cartesian2spherical(v_local[0,...], v_local[1,...], v_local[2,...], deg=True)
        elevs.append(elev)
        azims.append(azim)

    elevs_np = np.array(elevs)
    azims_np = np.array(azims)
    t = np.arange(0, len(elevs))

    # print(np.min(elevs_np))
    # print(np.max(elevs_np))
    # print(np.min(azims_np))
    # print(np.max(azims_np))

    # fig, ax = plt.subplots(2,2)
    # ax[0,0].plot(t[:1000], elevs_np[:1000])
    # ax[0,0].set_title('Elevations, SP')
    # ax[1,0].plot(t[:1000], azims_np[:1000])
    # ax[1,0].set_title('Azimuths, SP')
    # ax[0,1].plot(t[:1000], eph_df['sun_sublat'].values[:1000])
    # ax[0,1].set_title('Subsolar Latitude')
    # ax[1,1].plot(t[:1000], eph_df['sun_sublon'].values[:1000])
    # ax[1,1].set_title('Subsolar Longitude')
    # plt.show()

    # loop through ephemeris data and calculate illumination fraction for each day
    start = time.time()
    i = 0
    for _, row in eph_df.iterrows():

        if i % 100 == 0:
            print("On iteration " + str(i+1) + " / " + str(len(eph_df)))

        # Compute position of sun relative to local frames
        v_local = latlon2enu(row['sun_sublat'], row['sun_sublon'], row['sun_range'] * KM_AU * 1000, lats, lons, deg=True, esu=True)
        _, elev, azim = cartesian2spherical(v_local[0,...], v_local[1,...], v_local[2,...], deg=True)
        elev = elev.reshape((args.size, args.size))
        azim = azim.reshape((args.size, args.size))

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(elev, cmap='plasma')
        # ax[1].imshow(azim, cmap='plasma')
        # plt.show()

        # Pull elevations for this azimuth from the horizon db
        # Doing a linear interpolation between two nearest azimuths
        azim_low = np.floor(azim).astype(int)
        azim_high = np.ceil(azim).astype(int)
        azim_low[azim_low > 359] -= 360 # wraparound
        azim_high[azim_high > 359] -= 360
        horizon_elev_low = np.squeeze(np.take_along_axis(horizon_db, azim_low[...,None], axis=-1))
        horizon_elev_high = np.squeeze(np.take_along_axis(horizon_db, azim_high[...,None], axis=-1))
        horizon_elev = (horizon_elev_low + horizon_elev_high) / 2
        # print(horizon_elev.shape) # 800 x 800

        # Compare to the sun elevation and flag illuminated pixels
        # According to this paper: file:///home/margareh/Zotero/storage/QF8G24NM/S0019103514004278.html#s0020
        # the sun seen from the lunar horizon has an angular diameter of ~ 0.53 degrees
        # will use this to compute how much of the solar disk is visible
        # above and below the elevation from the ephemeris data
        sun_elev_low = elev - sun_rad_deg
        sun_elev_high = elev + sun_rad_deg

        # calculate area under the chord across the sun disk at average terrain elevation
        all_lit = (horizon_elev < sun_elev_low)
        all_dark = (horizon_elev > sun_elev_high)
        lower_disc = (sun_elev_low < horizon_elev) * (horizon_elev < elev)

        h = sun_elev_high - horizon_elev
        h[lower_disc] = horizon_elev[lower_disc] - sun_elev_low[lower_disc]

        # I ~could~ make this run without warning me about invalid values in arccos and sqrt
        # but where's the fun in that
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lit_area_degsq = sun_rad_sq * np.arccos(1 - (h / sun_rad_deg)) - (sun_rad_deg - h) * np.sqrt(sun_rad_sq - (sun_rad_deg - h)**2)
    
        lit_area_degsq[lower_disc] = sun_area_degsq - lit_area_degsq[lower_disc]
        lit_area_degsq[all_dark] = 0.0
        lit_area_degsq[all_lit] = sun_area_degsq

        # accumulate illumination fraction
        sum_illumin += lit_area_degsq / sun_area_degsq

        # print(np.min(sum_illumin))
        # print(np.max(sum_illumin))

        # # plot things
        # fig, ax = plt.subplots()
        # im = ax.imshow(sum_illumin, cmap='inferno')
        # fig.colorbar(im)
        # plt.show()

        i+= 1

    print("Elapsed time: %.2f" % (time.time() - start) + " s")

    # calculate illumination percentage
    illumin_frac = sum_illumin / len(eph_df)
    
    # also compute a PSR map (ish - technically PSRs are based on thermal info)
    psrs = illumin_frac < args.psr_threshold

    # store the illumination percentage array
    np.savez_compressed(os.path.join(args.outpath, "illumination_psrs.npz"),
             illumin_frac=illumin_frac,
             psrs=psrs)

    # plot the illumination percentage and PSR map
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(20,10)
    im = ax[0].imshow(illumin_frac, cmap='inferno')
    ax[1].imshow(psrs, cmap='binary')
    ax[0].set_title('Illumination (%)')
    ax[1].set_title('PSRs')
    fig.colorbar(im)

    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.outpath, "illumination_frac_and_psrs.png"), dpi=100, bbox_inches="tight")
        plt.close()

    return illumin_frac, psrs



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--eph', type=str, default='/media/ssd/ThesisWork/Volatiles/JPL Horizons/sun_position_2004_2024_values_only.txt', help='Ephemeris file')
    parser.add_argument('--horizons', type=str, default='/media/ssd/ThesisWork/Volatiles/processed/synthterrain/horizon', help='Directory with horizons files')
    parser.add_argument('--res', type=float, default=1., help='Resolution of grid')
    parser.add_argument('--size', type=int, default=800, help='Size of grid (dimension of one side)')
    parser.add_argument('--psr_threshold', type=float, default=0.0001, help='Threshold percentage below which pixels are classified as PSR')
    parser.add_argument('--outpath', type=str, default='/media/ssd/ThesisWork/Volatiles/processed/synthterrain/illumination', help='Path to where output should be stored')
    parser.add_argument('--solar_disc_angle', type=float, default=0.53, help='Diameter angle of solar disc from lunar surface')
    parser.add_argument('--plot', action='store_true', help='Flag for showing plot instead of saving')
    args = parser.parse_args()

    if os.path.exists(args.outpath) == False:
        os.mkdir(args.outpath)

    # run the main function
    main(args)
