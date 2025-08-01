# date:     7-31-2025   
# author:   margaret hansen
# purpose:  show horizons data

import os
import argparse
import imageio
import rasterio as rs
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection


# function to display data
def display_data(pth):

    plt.ion()
    fig, ax = plt.subplots()

    for i in range(360):

        # load the current dataset
        data = np.squeeze(np.load(os.path.join(pth, 'horizon_' + str(i) + '.npz'))['elevs'])

        # update the plot
        if i == 0:
            img = plt.imshow(data, cmap='Blues')
        else:
            img.set_data(data)
        ax.set_title("Azimuth: " + str(i))
        plt.pause(0.05)


# function to make and save images
def make_images(pth):

    if ~os.path.exists(os.path.join(pth, "images")):
        os.mkdir(os.path.join(pth, "images"))

    for i in range(360):

        fig, ax = plt.subplots()

        # load the current dataset
        data = np.squeeze(np.load(os.path.join(pth, 'horizon_' + str(i) + '.npz'))['elevs'])

        # add data to the plot and save
        plt.imshow(data, cmap='Blues')
        ax.set_title("Azimuth: " + str(i))
        plt.savefig(os.path.join(pth, "images/horizon_" + str(i) + ".png"), dpi=100, bbox_inches="tight")
        plt.close()

    # now make the gif of the images
    imgs = []
    for i in range(360):

        # load the image
        name = os.path.join(pth, "images/horizon_" + str(i) + ".png")
        img = imageio.imread(name)
        imgs.append(img)

    # save the output
    # duration is in ms
    imageio.mimsave(os.path.join(pth, "images/horizons.gif"), imgs, duration = 100, loop=0)


# function for interactive selection of points to generate horizons
def horizon_picker(args):

    # load heightmap
    hmap_f = rs.open(os.path.join(args.path, '../synthcraters_ex.tif'))
    hmap = hmap_f.read(1)

    # limit heightmap to border
    r = int(np.floor(args.max_range * 1000 / args.res))
    h,w = hmap.shape
    # y_pts = np.array([r, r, h-r, h-r, r])
    # x_pts = np.array([r, w-r, w-r, r, r])
    dem_data_limit = hmap[r:(h-r), r:(w-r)]
    hl, wl = dem_data_limit.shape

    del(hmap)
    hmap_f.close()
    
    # load elevation database
    horizon_db = np.zeros((hl, wl, 360))
    for i in range(360):
        elevs = np.squeeze(np.load(os.path.join(args.path, 'horizon_' + str(i) + '.npz'))['elevs'])
        horizon_db[...,i] = elevs

    del(elevs)

    # display 3 plots
    azims = np.arange(0, 360)

    # plt.ion()
    fig = plt.figure(layout="constrained")
    gs = GridSpec(2,2, figure=fig)
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])
    
    # add some titles
    ax1.set_title('Horizon profile for selected point')
    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('Elevation (degrees)')

    ax2.set_title('Terrain height (m)')
    ax3.set_title('Horizon elevation map for selected azimuth')

    # initial plots
    azim = 0
    pt = np.array([int(h/2), int(w/2)])
    heights = horizon_db[pt[1], pt[0], :]
    
    l1, = ax1.plot(azims, heights, c='tab:blue')
    l2 = ax1.axvline(azim, ymin=0, ymax=np.max(heights), c='gray', linestyle='dashed')
    ax1.set_ylim(np.min(horizon_db)-0.25, np.max(horizon_db)+0.25)

    ax2.imshow(dem_data_limit, cmap='terrain')
    pt1 = ax2.scatter(pt[0], pt[1], c='red', s=2)

    im1 = ax3.imshow(horizon_db[..., azim], cmap='Blues')
    pt2 = ax3.scatter(pt[0], pt[1], c='red', s=2)

    # hook for mouse selection of points in image
    def onclick(event):

        # switch between selections based on which plot was clicked
        curr_ax = event.inaxes
        if len(curr_ax.get_images()) > 0:
            
            x = event.xdata
            y = event.ydata

            # update points displayed in plot
            pt1.set_offsets(np.c_[x, y])
            pt2.set_offsets(np.c_[x, y])

            # update horizon line
            heights = horizon_db[int(y), int(x)]
            l1.set_data(azims, heights)
        
        elif len(curr_ax.get_lines()) > 0:

            # update the axis for the horizon elevation plot
            azim = event.xdata
            im1.set_data(horizon_db[..., int(azim)])
            l2.set_xdata([azim, azim])
        
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='display', help='Mode to use this tool in [display shows data, save creates images and a gif, interact allows for point selection]')
    parser.add_argument('--path', type=str, default='/media/ssd/ThesisWork/Volatiles/processed/synthterrain/horizon', help='Path to data')
    parser.add_argument('--max_range', type=float, default=0.1, help='Maximum range in km that horizons for which horizons were built (used to calculate border)')
    parser.add_argument('--res', type=float, default=1, help='Resolution of underlying DEM (used to calculate border)')
    args = parser.parse_args()

    # check mode
    args.mode = args.mode.lower()
    if args.mode not in ['display', 'save', 'interact']:
        args.mode = 'display'
        print("Incorrect mode chosen, defaulting to display")

    # perform action
    if args.mode == 'display':
        display_data(args.path)
    elif args.mode == 'save':
        make_images(args.path)
    elif args.mode == 'interact':
        horizon_picker(args)





