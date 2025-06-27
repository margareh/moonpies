"""
GP class and prediction function
"""

import numpy as np
import gpytorch

from torch import Tensor, no_grad, zeros
from gpytorch.settings import fast_pred_var

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


def gp_predict(gp, lik, pred_x, batch_size=None, gpu=False):

    # convert input to tensor if needed
    if not isinstance(pred_x, Tensor):
        pred_x = Tensor(pred_x)

    n = pred_x.shape[0]

    # send everything to GPU if desired
    if gpu:
        gp = gp.cuda()
        lik = lik.cuda()
        pred_x = pred_x.cuda()

    # set to eval mode
    gp.eval()
    lik.eval()

    if batch_size is not None:
        batches = int(np.ceil(n / batch_size))
        out_mean = zeros((n))
        out_var = zeros((n))    
        for b in range(batches):

            # print("On batch %d / %d" % (b+1, batches))
            
            # limit input data
            b_start = b*batch_size
            b_end = (b+1)*batch_size if (b+1)*batch_size <= n else n
            pred_x_b = pred_x[b_start:b_end,:]

            # predict from the gp
            with no_grad(), fast_pred_var():
                out = lik(gp(pred_x_b))

            # save mean and variance
            out_mean[b_start:b_end] = out.mean
            out_var[b_start:b_end] = out.variance
    
    else:
        with no_grad(), fast_pred_var():
            out = lik(gp(pred_x))
            out_mean = out.mean
            out_var = out.variance

    if gpu:
        out_mean = out_mean.cpu()
        out_var = out_var.cpu()

    return out_mean.numpy(), out_var.numpy()
