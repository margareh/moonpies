# date:     7-11-2025
# author:   margaret hansen
# purpose:  process Haworth DTM to get illumination amounts


import os
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rs


# load and display tif file
def load_tif(args):

    # load the tif file
    data = rs.open(args.path)
    
    # print some metadata about the file
    print(data.bounds) # geographic bounds
    print(data.width)
    print(data.height)
    print(data.count) # this is the number of bands
    print(data.crs) # projection EPSG code
    print(data.nodata) # value representing no data
    # 269XX = UTM north zone XX, NAD83
    # 326XX = UTM north zone XX, WGS 84
    # 4326 = LLA, WGS 84

    meta = {'bounds'    : data.bounds,
            'width'     : data.width,
            'height'    : data.height,
            'count'     : data.count,
            'crs'       : data.crs}

    # read the data and mask out missing values
    data_mask = data.read(1)
    data_mask[data_mask == data.nodata] = np.nan

    # either save or show a picture of it
    fname = args.path.replace('.tif', '.png')
    ext = [data.bounds[0], data.bounds[2], data.bounds[1], data.bounds[3]]
    plt.imshow(data_mask, cmap='gray', extent=ext)
    if args.plot:
        plt.show()
    else:
        plt.savefig(fname, dpi=100, bbox_inches='tight')

    # close the dataset
    data.close()

    # return the loaded data and associated metadata
    return data_mask, meta


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/media/ssd/ThesisWork/Volatiles/SouthPoleData/Haworth_DEM_1mpp/Lunar_LROnac_Haworth_sfs-dem_1m_v3.tif', help='Path to tif file')
    parser.add_argument('--plot', action='store_true', help='Flag to plot tif file')
    args = parser.parse_args()

    # load the data
    data, meta = load_tif(args)

    # process the data


