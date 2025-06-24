"""
GP for interpolating between mixing ratios and ejecta temps
"""

import pandas as pd
import numpy as np

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
    return xx, yy, zz


# fit a GP
def fit_gp(xx, yy, zz, iters=1000):

    # set up data
    train_x = np.vstack(xx, yy)

    # set up model
    lik = gpytorch.likelihoods.GaussianLikelihood()
    mod = GP(train_x, zz, lik)
    lik.train()
    mod.train()

    # set up optimizer
    opt = torch.optim.Adam(mod.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, mod)

    # iterate
    for i in range(iters):
        opt.zero_grad()
        out = mod(train_x)
        loss = -mll(out, zz)
        loss.backward()
        opt.step()

    return mod



if __name__ == "__main__":

    # load the data
    xx, yy, zz = load_data('/home/margareh/moonpies/moonpies/data/ballistic_sed_frac_melted_mean.csv')

    # fit the GP
    gp = fit_gp(xx, yy, zz, iters=1000)

    # save the GP model
    torch.save(gp.state_dict(), '/home/margareh/moonpies/moonpies/data/ballistic_sed_frac_melted_mean_gp.pth')

    # show some interpolation plots

