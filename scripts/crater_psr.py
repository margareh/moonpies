# compare tifs with crater locations and radii for moonpies

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rs


if __name__ == "__main__":

    # load tif
    psr_f = rs.open('/media/ssd/ThesisWork/Volatiles/SouthPoleData/LPSR_80S_20MPP_ADJ.TIF')
    psr = psr_f.read(1)
    
    # downsample for display
    r = 50
    n = int(psr.shape[0] / r)
    psr_sm = psr.reshape((n, r, n, r)).sum(axis=1).sum(axis=2)
    # print(psr_sm.shape) # 608 x 608

    # load crater and basin data
    crater_cols: tuple = ('cname', 'lat', 'lon', 'psr_lat', 'psr_lon', 'diam', 
                        'age', 'age_low','age_upp', 'psr_area', 'age_ref', 
                        'prio', 'notes')
    df_crater = pd.read_csv('~/moonpies/moonpies/data/crater_list.csv', names=crater_cols, header=0)

    basin_cols: tuple = ('cname', 'lat', 'lon', 'diam', 'inner_ring_diam', 
                         'bouger_diam', 'age', 'age_low', 'age_upp', 'ref')
    df_basin = pd.read_csv('~/moonpies/moonpies/data/basin_list.csv', names=basin_cols, header=0)

    # compute radius in meters
    df_crater['rad'] = df_crater['diam'] * 1000 / 2
    df_basin['rad'] = df_basin['diam'] * 1000 / 2

    n_crater = len(df_crater)
    n_basin = len(df_basin)

    df = pd.concat([df_crater, df_basin])
    # print(len(df)) # 51

    # grid associated with psr data
    ysize = (psr_sm.shape[0] / 2) * r * 20
    xsize = (psr_sm.shape[1] / 2) * r * 20
    yrange = np.arange(ysize, -ysize, -r*20)
    xrange = np.arange(-xsize, xsize, r*20)
    yy, xx = np.meshgrid(yrange, xrange, indexing='ij')
    # print(yy.shape) # 608 x 608
    # print(xx.shape) # 608 x 608

    # compute distance from grid to craters
    crater_dists_all = np.zeros((len(df), yy.shape[0], xx.shape[1]))
    crater_mask = np.zeros((len(df), psr_sm.shape[0], psr_sm.shape[1]))
    # print(crater_dists_all.shape) # 51 x 608 x 608
    # print(crater_mask.shape) # 51 x 608 x 608

    rp=1737.4e3
    z = np.sqrt(rp**2 - xx**2 - yy**2)
    grdlat = np.rad2deg(-np.arcsin(z / rp))
    grdlon = np.rad2deg(np.arctan2(xx, yy))

    # print(np.min(grdlat)) # -90
    # print(np.max(grdlat)) # -75.67
    # print(np.min(grdlon)) # -179
    # print(np.max(grdlon)) # 180
    
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(grdlat)
    # ax[1].imshow(grdlon)
    # plt.show()
    # plt.close()

    cr_id = 0
    for i, row in df.iterrows():

        # compute distances
        clon, clat, crad = row[['lon', 'lat', 'rad']]
        # print(row[['lon', 'lat', 'rad']])
        lon1, lat1, lon2, lat2 = map(np.deg2rad, [clon, clat, grdlon, grdlat])
        sin2_dlon = np.sin((lon2 - lon1) / 2) ** 2
        sin2_dlat = np.sin((lat2 - lat1) / 2) ** 2
        a = sin2_dlat + np.cos(lat1) * np.cos(lat2) * sin2_dlon
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        crater_dists = rp * c
        # print(crater_dists.shape) # 608 x 608

        # print(np.min(crater_dists))
        # print(np.max(crater_dists))
        # plt.imshow(crater_dists)
        # plt.show()
        # plt.close()

        # create crater masks
        crater_mask[cr_id] = (crater_dists <= crad)
        crater_dists_all[cr_id] = copy.copy(crater_dists)
        cr_id += 1

    crater_mask_all = np.any(crater_mask[0:n_crater], axis=0)
    basin_mask_all = np.any(crater_mask[n_crater:], axis=0)
    # print(crater_mask_all.shape) # 608 x 608

    # compare
    ext = [-xsize, xsize, -ysize, ysize] # lrbt

    fig, ax = plt.subplots()
    ax.imshow(psr_sm, cmap='binary', extent=ext)
    ax.imshow(crater_mask_all, cmap='Oranges', alpha=0.5, extent=ext)
    ax.imshow(basin_mask_all, cmap='Blues', alpha=0.5, extent=ext)
    plt.show()


