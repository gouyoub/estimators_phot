import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import scipy.interpolate as interp
import time
import os

def spv3_ell_binning(ell_spv3):
    ell_edges_base = np.logspace(np.log10(10), np.log10(5000), 33)
    nbl = ell_spv3.size
    elli, ellf = np.zeros(nbl, int), np.zeros(nbl, int)
    for i in range(nbl):
        elli[i] = round(ell_edges_base[i])
        ellf[i] = round(ell_edges_base[i+1])
    return elli, ellf

def linear_lmin_binning(NSIDE, lmin, bw):
    lmax = 2*NSIDE
    nbl = (lmax-lmin)//bw + 1
    elli = np.zeros(nbl, int)
    elle = np.zeros(nbl, int)

    for i in range(nbl):
        elli[i] = lmin + i*bw
        elle[i] = lmin + (i+1)*bw

    b = nmt.NmtBin.from_edges(elli, elle)
    return b

def coupling_matrix(bin_scheme, mask, wkspce_name):
    print('Compute the mixing matrix')
    start = time.time()
    fmask = nmt.NmtField(mask, [mask]) # nmt field with only the mask
    w = nmt.NmtWorkspace()
    if os.path.isfile(wkspce_name):
        print('Mixing matrix has already been calculated and is in the workspace file : ', wkspce_name, '. Read it.')
        w.read_from(wkspce_name)
    else :
        print('The file : ', wkspce_name, ' does not exists. Calculating the mixing matrix and writing it.')
        w.compute_coupling_matrix(fmask, fmask, bin_scheme)
        w.write_to(wkspce_name)
    print('Done computing the mixing matrix. It took ', time.time()-start, 's.')
    return w

def edges_log_binning(NSIDE, lmin, nbl):
    lmax = 2*NSIDE
    bin_edges = np.logspace(np.log10(lmin), np.log10(lmax), nbl)
    bin_edges = np.floor(bin_edges).astype(int)
    b = nmt.NmtBin.from_edges(bin_edges[:-1], bin_edges[1:])
    return b

# def nmt_workspace(mask, binning, w_fname):
#     w = nmt.NmtWorkspace()
#     fmask = nmt.NmtField(mask, [mask]) # nmt field with only the mask
#     w.compute_coupling_matrix(fmask, fmask, binning) # compute the mixing matrix (which only depends on the mask) just once
#     w.write_to(w_fname)