# compare results from two moonpies runs

import os
import argparse
import numpy as np
import pandas as pd


# compare to csv files
def compare_csv(csv1, csv2):
    
    # load both files
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # compare the two
    comp = df1.compare(df2)
    print(comp)


# compare two numpy files
def compare_npy(npy1, npy2):

    # load both files
    data1 = np.load(npy1)
    data2 = np.load(npy2)

    # compare the two
    compare = (data1 - data2)
    print(np.nansum(compare))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, help='Path to first set of results')
    parser.add_argument('--path2', type=str, help='Path to second set of results')
    args = parser.parse_args()

    # get list of csv files from both paths
    csv_list1 = [f for f in os.listdir(args.path1) if f.find('csv') >= 0]
    csv_list2 = [f for f in os.listdir(args.path2) if f.find('csv') >= 0]
    csv_list1.sort()
    csv_list2.sort()
    # print(csv_list1)
    # print(csv_list2)

    if len(csv_list1) != len(csv_list2):
        print('Error: mismatching lists of csv files')
    else:
        for i in range(len(csv_list1)):
            print(csv_list1[i])
            f1 = os.path.join(args.path1, csv_list1[i])
            f2 = os.path.join(args.path2, csv_list2[i])
            compare_csv(f1, f2)
    
    # get list of numpy files
    npy_list1 = [f for f in os.listdir(args.path1) if f.find('npy') >= 0]
    npy_list2 = [f for f in os.listdir(args.path2) if f.find('npy') >= 0]
    npy_list1.sort()
    npy_list2.sort()
    # print(npy_list1)
    # print(npy_list2)

    if len(npy_list1) != len(npy_list2):
        print('Error: mismatching lists of npy files')
    elif len(npy_list1) > 0:
        for i in range(len(npy_list1)):
            print(npy_list1[i])
            f1 = os.path.join(args.path1, npy_list1[i])
            f2 = os.path.join(args.path2, npy_list2[i])
            compare_npy(f1, f2)

