import torch as pt
from torch.autograd import Function
import numpy as np
import exp

class ExpR(Function):  # exponential representation function

    @staticmethod
    def forward(ctx, data, eta, kappa):
        feat, kappa_eta, backward_data = exp.calc_feat(data[0], data[1], eta.cpu().detach().numpy(), kappa.cpu().detach().numpy())
        ctx.kappa_eta = kappa_eta
        ctx.backward_data = backward_data
        ctx.spar_ind = data[1]
        return pt.tensor(feat)

    @staticmethod
    def backward(ctx, grad_data):
        eta_deriv, kappa_deriv = exp.calc_deriv(ctx.kappa_eta, ctx.backward_data, ctx.spar_ind, grad_data.cpu().detach().numpy())
        return None, pt.tensor(eta_deriv), pt.tensor(kappa_deriv)
