# date:     8-5-2025
# author:   margaret hansen
# purpose:  format crater list from synthterrain in moonpies format

import os
import argparse
import copy
import numpy as np
import pandas as pd


if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/media/ssd/ThesisWork/Volatiles/processed/synthterrain', help='Path to crater and illumination models')
    parser.add_argument('--outpath', type=str, default='../moonpies/data', help='Output folder to save results in')
    args = parser.parse_args()

    # load crater data
    craters = pd.read_csv(os.path.join(args.datapath, 'synthcraters_ex.csv'), names=['x', 'y', 'diameter', 'age', 'd/D'], header=0)
    print(len(craters)) # 29174

    # load psrs
    psrs = np.load(os.path.join(args.datapath, 'illumination/illumination_psrs.npz'))

    # compute area of PSRs associated with each crater
    # currently estimating them based on diameter
    craters['psr_area'] = (0.9 / 4) * (np.pi * craters['diameter'] ** 2)

    # age in Ga
    craters['age'] /= 1e9

    # lower and upper bound on age
    craters['age_low'] = copy.copy(craters['age'])
    craters['age_upp'] = copy.copy(craters['age'])

    # filter to craters within bounding box of PSR map
    cond = (craters['x'] > 100) * (craters['x'] < 900) * (craters['y'] > 100) * (craters['y'] < 900)
    craters_limit = craters.loc[cond]
    print(len(craters_limit)) # 18831

    # cname (index)
    craters_limit['cname'] = craters_limit.index

    # resave psrs in correct folder
    np.savez(os.path.join(args.outpath, 'psrs.npz'), psrs=psrs)

    # save crater data
    craters_limit.to_csv(os.path.join(args.outpath, 'synthterrain_craters.csv'), header=True, index=False)
