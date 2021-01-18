import numpy as np
import torch as pt
from torch.autograd import Function
import plin

class PlinR(Function):  # piecewise linear function
    @staticmethod
    def forward(ctx, data, eta, kappa):
        feat, backward_data = plin.calc_feat(data[0], data[1], eta.cpu().detach().numpy(), kappa.cpu().detach().numpy())
        ctx.backward_data = backward_data
        ctx.spar_ind = data[1]
        return pt.tensor(feat)

    @staticmethod
    def backward(ctx, grad_data):
        eta_deriv, kappa_deriv = plin.calc_deriv(ctx.backward_data, ctx.spar_ind, grad_data.cpu().detach().numpy())
        return None, pt.tensor(eta_deriv), pt.tensor(kappa_deriv)
