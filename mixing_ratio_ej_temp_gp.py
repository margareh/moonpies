"""
GP for interpolating between mixing ratios and ejecta temps
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gpytorch
import torch

from moonpies import config
from moonpies.utils.utils import get_grid_arrays
from moonpies.utils.load_data import read_crater_list, read_basin_list
from moonpies.utils.save_output import get_gc_dist_grid
from moonpies.processes.ballistic import get_mixing_ratio_oberbeck, ejecta_temp


# GP class for exact estimation
class GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# load the data from a provided file name
def load_data(file):
    data = pd.read_csv(file, index_col=0)
    x = data.columns.to_numpy()[1:].astype(np.float64)
    y = data.index.to_numpy()[1:].astype(np.float64)
    nx = len(x)
    ny = len(y)
    xx, yy = np.meshgrid(x, y) # these have shape nx x ny
    xx = xx.reshape((nx*ny))
    yy = yy.reshape((nx*ny))
    # skip the first row/column because they're all the same values
    zz = data.values[1:,1:].astype(np.float64).reshape((nx*ny))
    return xx, yy, zz, nx, ny


# fit a GP
def fit_gp(xx, yy, zz, iters=1000):

    # set up data
    train_x = np.vstack((xx, yy)).T
    train_x_t = torch.Tensor(train_x)
    train_z_t = torch.Tensor(zz)

    # set up model
    lik = gpytorch.likelihoods.GaussianLikelihood()
    mod = GP(train_x_t, train_z_t, lik)
    lik.train()
    mod.train()

    # set up optimizer
    opt = torch.optim.Adam(mod.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, mod)

    # iterate
    for i in range(iters):
        opt.zero_grad()
        out = mod(train_x_t)
        loss = -mll(out, train_z_t)
        loss.backward()
        if i % 100 == 0:
            print("Iter %d: loss = %5.4f" % (i, loss))
        opt.step()

    return mod, lik


# plot some things from the GP
def plot_gp(xx, yy, zz, gp_mod, gp_lik, nx, ny):

    gp_mod.eval()
    gp_lik.eval()

    # predict from the gp over the training set
    train_x = torch.Tensor(np.vstack((xx, yy)).T)
    # print(train_x.shape) # 3120 x 2 --> should be fine predicting over 1000 x 2 inputs?
    with torch.no_grad():
        zz_pred = gp_lik(gp_mod(train_x))
    zz_mean = zz_pred.mean.numpy()
    zz_var = zz_pred.variance.numpy()
    error = (zz_mean - zz)**2
    print("MSE (Var SE): %5.4f (%5.4f)" % (np.mean(error), np.var(error)))

    # create plots
    fig, ax = plt.subplots(2, 2, figsize=(20,20))

    # actual and error
    im1 = ax[0,0].imshow(zz.reshape((nx, ny), order='F').T, cmap='plasma')
    im2 = ax[0,1].imshow(error.reshape((nx, ny), order='F').T, cmap='coolwarm')
    ax[0,0].set_title('Actual')
    ax[0,1].set_title('Error')
    fig.colorbar(im1, ax=ax[0,0])
    fig.colorbar(im2, ax=ax[0,1])

    # predicted mean and uncertainty
    im3 = ax[1,0].imshow(zz_mean.reshape((nx, ny), order='F').T, cmap='plasma')
    im4 = ax[1,1].imshow(zz_var.reshape((nx, ny), order='F').T, cmap='Reds')
    ax[1,0].set_title('Predicted Mean')
    ax[1,1].set_title('Predicted Uncertainty')
    fig.colorbar(im3, ax=ax[1,0])
    fig.colorbar(im4, ax=ax[1,1])

    # plt.show()
    plt.savefig('/home/margareh/moonpies/moonpies/data/gp_preds.png', dpi=100, bbox_inches='tight')
    plt.close()


def load_pred_data(cfg):

    grdy, grdx = get_grid_arrays(cfg)

    df_craters = read_crater_list(cfg)
    df_craters["isbasin"] = False
    df_basins = read_basin_list(cfg)
    df_basins["isbasin"] = True
    df = pd.concat([df_craters, df_basins])

    dists = get_gc_dist_grid(df, grdx, grdy, cfg, mask=True)
    mixing_ratio = get_mixing_ratio_oberbeck(dists, cfg)
    ej_temp = ejecta_temp(df, cfg)

    return mixing_ratio, ej_temp


# predict over new inputs
def predict_gp(gp_mod, gp_lik, mixing_ratio, ej_temp, gpu=False, batch_size=None):

    # loop through craters and predict for each
    # print(mixing_ratio.shape) # 51 x 608 x 608
    n = mixing_ratio.shape[-1]
    c = mixing_ratio.shape[0]
    out_all = np.zeros((c,n*n))
    for i in range(c):
        print("Crater %d / %d" % (i+1, c))

        # limit to current crater
        ej_temp_c = ej_temp[i]
        mix_r_c = mixing_ratio[i,...]
        pred_x = np.dstack((np.ones_like(mix_r_c) * ej_temp_c, mix_r_c)).reshape((n*n,2))
        pred_x = torch.from_numpy(pred_x)
        # print(pred_x.shape) # 369664 x 2

        if gpu:
            gp_mod = gp_mod.cuda()
            gp_lik = gp_lik.cuda()
            pred_x = pred_x.cuda()

        # make sure models are in eval mode
        gp_mod.eval()
        gp_lik.eval()

        batch_size = n*n if batch_size is None else batch_size
        num_batches = int(np.ceil(n*n / batch_size))
        # print(batch_size)
        # print(num_batches)
        for b in range(num_batches):
            print("Batch %d / %d" % (b+1, num_batches))

            # further limit the data
            b_start = b*batch_size
            b_end = (b+1)*batch_size if (b+1)*batch_size <= n*n else n*n
            pred_x_b = pred_x[b_start:b_end,:]

            # predict
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                out = gp_lik(gp_mod(pred_x_b))

            out_mean = out.mean
            if gpu:
                out_mean = out_mean.cpu()
            out_all[i, b_start:b_end] = out_mean.numpy()
        
    melt_frac = out_all.reshape((c,n,n))

    # save the output
    np.savez('/home/margareh/moonpies/data/pred_melt_frac_gp.npz', melt_frac=melt_frac)



if __name__ == "__main__":

    # file paths
    inpath = '/home/margareh/moonpies/moonpies/data/ballistic_sed_frac_melted_mean.csv'
    outpath = '/home/margareh/moonpies/moonpies/data/ballistic_sed_frac_melted_mean_gp.pth'

    # load the data
    xx, yy, zz, nx, ny = load_data(inpath)

    if os.path.exists(outpath) == False:

        # fit the GP
        gp_mod, gp_lik = fit_gp(xx, yy, zz, iters=1000)

        # save the GP model
        torch.save(gp_mod.state_dict(), outpath)
    
    else:

        # load the model
        state_dict = torch.load(outpath)
        gp_lik = gpytorch.likelihoods.GaussianLikelihood()
        train_x = np.vstack((xx, yy)).T
        train_x_t = torch.Tensor(train_x)
        train_z_t = torch.Tensor(zz)
        gp_mod = GP(train_x_t, train_z_t, gp_lik)
        gp_mod.load_state_dict(state_dict)

    # plot some things
    plot_gp(xx, yy, zz, gp_mod, gp_lik, nx, ny)

    # now pre-compute output for all times and data points for the model
    cfg = config.Cfg()
    mixing_ratio, ej_temp = load_pred_data(cfg)
    predict_gp(gp_mod, gp_lik, mixing_ratio, ej_temp, gpu=True, batch_size=1000)

