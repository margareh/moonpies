# date:     7-9-2025
# author:   margaret hansen
# purpose:  make gif from images saved from moonpies run

import os
import argparse
import numpy as np
import imageio


def make_gifs(args):

    # get file paths
    dirs = os.listdir(args.path)
    dirs_z = [f.zfill(10) for f in dirs if f.find('gif') < 0]
    dirs_z.sort(reverse=True)

    # loop through file paths and make a gif for each type of image
    for i in args.images:
        print("Saving gif for " + i)
        imgs = []
        sizes = []
        for f in dirs_z:
            name = os.path.join(args.path, str(int(f)), i)
            new_img = imageio.imread(name)
            imgs.append(new_img)
            sizes.append(new_img.shape)
            print(new_img.shape)

        # resize if needed
        sizes_np = np.array(sizes)
        sizes_unique = np.unique(sizes_np, axis=0)
        if sizes_unique.shape[0] > 1:
            min_h = np.min(sizes_unique[:,0])
            min_w = np.min(sizes_unique[:,1])
            min_c = np.min(sizes_unique[:,2])
            imgs_new = [i[0:min_h, 0:min_w, 0:min_c] for i in imgs]
        else:
            imgs_new = imgs

        # save the output
        # duration is in ms
        imageio.mimsave(os.path.join(args.path, i.replace('png', 'gif')), imgs_new, duration = 500, loop=0, palettesize=256)


# make gifs out of numpy files ordered from 0 onwards
def make_gifs_np(args):

    # get file paths
    dirs = os.listdir(args.path)
    dirs_z = [f for f in dirs if f.find('gif') < 0]
    dirs_z.sort()
    # print(dirs_z)

    # loop through file paths and make a gif for each type of image
    imgs = []
    for f in dirs_z:
        
        # load the file
        data = np.squeeze(np.load(os.path.join(args.path, f))['elevs'], axis=2)
        # print(data.shape)
        imgs.append(data)

    img_arr = np.array(imgs)
    # print(img_arr.shape) # 360 x 800 x 800
    minval = np.nanmin(img_arr)
    maxval = np.nanmax(img_arr)
    print(minval)
    print(maxval)

    # normalize the data to [0,1]
    img_norm = (img_arr - minval) / (maxval - minval)
    img_list = list(img_norm)
    # print(len(img_list)) # 360

    fname = dirs_z[0].replace('_0.npz', '.gif')
    imageio.mimsave(os.path.join(args.path, fname), img_list, duration = 500, loop=0)


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/media/ssd/ThesisWork/Volatiles/SimData/81082', help='Path to output files')
    parser.add_argument('--images', type=str, nargs='+', default=['craters_and_psrs.png', 'ice_and_ejecta.png', 'ice_depth_and_fraction.png'])
    parser.add_argument('--numpy', action='store_true', help='Flag for whether data is in npz format or png format')
    args = parser.parse_args()

    # call main function
    if args.numpy:
        make_gifs_np(args)
    else:
        make_gifs(args)
