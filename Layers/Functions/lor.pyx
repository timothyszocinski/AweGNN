import numpy as np
cimport cython
from numba_lor import calc_elem_spec_feat, calc_elem_spec_deriv

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_feat(list data, int[:, :] spar_ind, double[:] ns, double[:] ks):
    cdef int num_samp = len(data)
    init_feat = np.zeros((num_samp, 400), dtype=np.float64)
    cdef double[:, :] feat = init_feat
    init_kappa_eta = np.zeros(100, dtype=np.float64)
    cdef double[:] kappa_eta = init_kappa_eta
    cdef list backward_data = []
    cdef list elem_spec_list
    cdef Py_ssize_t i, j
    for i in range(num_samp):
        elem_spec_list = []
        for j in range(100):
            if spar_ind[i, j] == -1:
                break
            (feat[i, 4*spar_ind[i, j]], feat[i, 4*spar_ind[i, j]+1], feat[i, 4*spar_ind[i, j]+2], feat[i, 4*spar_ind[i, j]+3],
            E, X_0, X_1, X_act) = calc_elem_spec_feat(data[i][j][0], data[i][j][1], ns[spar_ind[i, j]], ks[spar_ind[i, j]])
            elem_spec_list.append((E, X_0, X_1, X_act))
        backward_data.append(elem_spec_list)
    for j in range(100):
        kappa_eta[j] = ks[j] / ns[j]
    return feat, kappa_eta, backward_data

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_deriv(double[:] kappa_eta, list backward_data, int[:, :] spar_ind, double[:, :] grad_data):
    cdef int num_samples = len(backward_data)
    kappa_deriv_init = np.zeros(100, dtype=np.float64)
    cdef double[:] kappa_deriv = kappa_deriv_init
    eta_deriv_init = np.zeros(100, dtype=np.float64)
    cdef double[:] eta_deriv = eta_deriv_init
    cdef Py_ssize_t i, j
    cdef double g0
    cdef double g1
    for i in range(num_samples):
        for j in range(100):
            if spar_ind[i, j] == -1:
                break
            g0, g1 = calc_elem_spec_deriv(backward_data[i][j][0], backward_data[i][j][1],
            backward_data[i][j][2], backward_data[i][j][3], kappa_eta[spar_ind[i, j]],
            (grad_data[i, 4*spar_ind[i, j]], grad_data[i, 4*spar_ind[i, j]+1], grad_data[i, 4*spar_ind[i, j]+2], grad_data[i, 4*spar_ind[i, j]+3]))
            eta_deriv[spar_ind[i, j]] += g0
            kappa_deriv[spar_ind[i, j]] += g1
    return eta_deriv, kappa_deriv
