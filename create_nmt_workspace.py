""" ChatGPT generated header
Script Header:
This script demonstrates the computation of a coupling matrix for pseudo-Cl power spectrum estimation
using the Namaster library with linear binning scheme generation.

Author: Sylvain Gouyou Beauchamps

Date: 02/29/2024

Description:
This script reads a binary mask file defining the regions of the sky, generates a linear binning
scheme with a minimum multipole of 10 and a bin width of 5, computes the coupling matrix using
Namaster's NmtField and NmtWorkspace objects, and writes the workspace to a file. This workspace
can then be used to compute the 3x2pt partial-sky covariance matrix.

Dependencies:
- Python 3.x
- healpy
- Namaster library
- utils.py module (located in the '../lib' directory)

Usage:
python workspace_from_mask.py

Inputs:
- '../input/fullsky_mask_binary_NS32.fits': Input binary mask file.
- '../input/fullsky_NmtWorkspace_NS32_LMIN10_BW5.fits': Output file for the computed coupling matrix.

Outputs:
- The computed coupling matrix is saved in the specified output file.

"""
import healpy as hp
import anglib as al
import pymaster as nmt
import numpy as np

# Load mask file
mask_fname = '/Users/sgouyoub/Documents/work/euclid/nlbias/data/masks/30deg_binary.fits'
mask = hp.read_map(mask_fname)

# Compute NSIDE from mask
NSIDE = hp.npix2nside(mask.size)
print(NSIDE)
# Generate linear binning scheme
# binning = al.edges_log_binning(NSIDE=NSIDE, lmin=10, nbl=11)
# binning = al.edges_binning(NSIDE=NSIDE, lmin=10, bw=50)
lmax=100
# binning = al.log_binning(lmax=lmax, lmin=10, nbl=32)
binning = al.linear_binning(lmax=lmax, lmin=2, bw=1)
# binning = nmt.NmtBin.from_nside_linear(NSIDE, nlb=50, is_Dell=False)

# lmin =  10
# lmax = 1024
# ell_bins = 32
# bpw_edges = np.logspace(np.log10(lmin), np.log10(lmax), ell_bins, dtype=int)
# binning = nmt.NmtBin.from_edges(ell_ini=bpw_edges[:-1], ell_end=bpw_edges[1:])

print(binning.get_effective_ells())

# Compute coupling matrix and save to file
w_fname = f'/Users/sgouyoub/Documents/work/euclid/3x2/data/nmt_workspace/FS2_octant_WL_6EP_noIA-nomag-nored-nowh_sharpcut_shapenoise_coupled_NmtWorkspace_NS{NSIDE}_LBINlin_LMAX{lmax}_LMIN2_BW1_galaxy_cl.fits'
w = al.coupling_matrix(binning, mask, w_fname, 0, 0)

# Compute coupling matrix and save to file
# w_fname = f'/Users/sgouyoub/Documents/work/euclid/3x2/data/nmt_workspace/30deg_apo05_NS{NSIDE}_LBINlog_LMAX{lmax}_LMIN10_NELL32_galaxy_shear_cl.fits'
# w = al.coupling_matrix(binning, mask, w_fname, 0, 2)

# # Compute coupling matrix and save to file
# w_fname = f'/Users/sgouyoub/Documents/work/euclid/3x2/data/nmt_workspace/30deg_apo05_NS{NSIDE}_LBINlog_LMAX{lmax}_LMIN10_NELL32_shear_cl.fits'
# w = al.coupling_matrix(binning, mask, w_fname, 2, 2)
