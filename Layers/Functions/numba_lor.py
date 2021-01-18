from numba import njit
import numpy as np

@njit(fastmath=True, cache=True)
def calc_elem_spec_feat(dist, E, n, k):
    num = dist.shape[0] * dist.shape[1]
    X_0 = dist / n
    X_1 = X_0**k
    X_act = np.where(dist > 0, 1 / (1 + X_1), 0)
    act_sum = np.sum(X_act)
    elec_act_sum = np.sum(X_act * E)
    return act_sum, elec_act_sum, act_sum / num, elec_act_sum / num, E, X_0, X_1, X_act

@njit(fastmath=True, cache=True)
def calc_elem_spec_deriv(E, X_0, X_1, X_act, kappa_eta, grad_data):
    num = X_0.shape[0] * X_0.shape[1]
    dL = grad_data[0] + grad_data[2] / num
    dL_elec = grad_data[1] + grad_data[3] / num
    eta = X_1 * X_act**2
    eta_elec = eta * E
    kappa = - eta * np.log(X_0 + 1e-12)
    kappa_elec = kappa * E
    eta_deriv = np.sum(eta)
    elec_eta_deriv = np.sum(eta_elec)
    kappa_deriv = np.sum(kappa)
    elec_kappa_deriv = np.sum(kappa_elec)
    return kappa_eta*(eta_deriv*dL+elec_eta_deriv*dL_elec), kappa_deriv*dL+elec_kappa_deriv*dL_elec
