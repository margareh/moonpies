"""
GP class and prediction function
"""

import gpytorch

from torch import Tensor, no_grad

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


def gp_predict(gp, lik, pred_x):

    # convert input to tensor if needed
    if not isinstance(pred_x, Tensor):
        pred_x = Tensor(pred_x)

    # predict from the gp
    with no_grad():
        out = lik(gp(pred_x))

    # reshape mean and variance from output
    out_mean = out.mean.numpy()
    out_var = out.variance.numpy()
