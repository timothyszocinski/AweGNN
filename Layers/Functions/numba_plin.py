from numba import njit
import numpy as np

@njit(fastmath=True, cache=True)
def calc_elem_spec_feat(dist, E, n, k):
    num = dist.shape[0] * dist.shape[1]
    ex = np.exp(-k)
    slope =  - 0.5 * ex / n
    X = np.where((dist > 2 * n) | (dist == 0), 0, 1 + slope * dist)
    X = np.where((n < dist) & (dist < 2 * n), ex + slope * dist, X)
    act_sum = np.sum(X)
    elec_act_sum = np.sum(X * E)
    return act_sum, elec_act_sum, act_sum / num, elec_act_sum / num, dist, E, n, k

@njit(fastmath=True, cache=True)
def calc_elem_spec_deriv(dist, E, n, k, grad_data):
    num = dist.shape[0] * dist.shape[1]
    dL = grad_data[0] + grad_data[2] / num
    dL_elec = grad_data[1] + grad_data[3] / num
    ex = np.exp(-k)
    common_term = 0.5 * ex / n
    eta = np.where((dist > 2 * n) | (dist == 0), 0, common_term * dist / n)
    eta_elec = eta * E
    kappa = np.where((dist > 2 * n) | (dist == 0), 0, common_term * dist)
    kappa = np.where((n < dist) & (dist < 2 * n), - ex + kappa, kappa)
    kappa_elec = kappa * E
    eta_deriv = np.sum(eta)
    elec_eta_deriv = np.sum(eta_elec)
    kappa_deriv = np.sum(kappa)
    elec_kappa_deriv = np.sum(kappa_elec)
    return eta_deriv*dL+elec_eta_deriv*dL_elec, kappa_deriv*dL+elec_kappa_deriv*dL_elec
