"""
GP for interpolating between mixing ratios and ejecta temps
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gpytorch
import torch


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
    x = data.columns.to_numpy().astype(np.float64)
    y = data.index.to_numpy().astype(np.float64)
    nx = len(x)
    ny = len(y)
    xx, yy = np.meshgrid(x, y) # these have shape nx x ny
    xx = xx.reshape((nx*ny))
    yy = yy.reshape((nx*ny))
    zz = data.values.astype(np.float64).reshape((nx*ny))
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

