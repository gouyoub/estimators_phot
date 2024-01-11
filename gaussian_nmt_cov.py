import numpy as np
import pymaster as nmt
import healpy as hp
import argparse
import itertools
import time

def get_cosmosis_Cl(ref, nbins, symmetric):
    cl = {} 
    keys = []  
    for i in range(nbins):
        if symmetric: j0 = i 
        else: j0 = 0 
        for j in range(j0,nbins):
            k = '-'.join([str(j+1), str(i+1)])
            keys.append(k)
            cl[k] = np.loadtxt(ref+'bin_'+str(j+1)+'_'+str(i+1)+'.txt')
    return cl, keys

#-- Argument parser
parser = argparse.ArgumentParser(description='Compute Gaussian covariance with a mask.')
parser.add_argument('-m', '--mask', dest='mask_fits', type=str, required=True, help='Absolute path to mask fits file.')
parser.add_argument('-c', '--cell', dest='ref_cell', type=str, required=True, help='Absolute path to the directory containing\
    the theory c_ell in the CosmoSIS text format. These Cls should be the unbinned ones for all ell values !')
parser.add_argument('-w', '--workspace', dest='workspace_fits', type=str, required=True, help='Absolute path to a NaMaster workspace fits file.')
parser.add_argument('-o', '--output', dest='output_name', type=str, required=True, help='Absolute path to saved output.')
parser.add_argument('-n', '--nzbins', dest='nzbins', type=int, required=True, help='Number of redshift bins.')
parser.add_argument('--auto_only', dest='auto_only', help='To compute the cov for the auto-cl only.', action='store_true')
parser.add_argument('--symmetric', dest='symmetric', help='If the Cl(zi,zj) = Cl(zj,zi).', action='store_true')

args = parser.parse_args()
dictargs = vars(args)
print("-----------------**  ARGUMENTS  **------------------------", flush=True)
for keys in dictargs.keys():
    print(keys, " = ", dictargs[keys], flush=True)

#-- Arguments
mask_fits = args.mask_fits
ref_cell = args.ref_cell
workspace_fits = args.workspace_fits
output_name = args.output_name
nzbins = args.nzbins
auto_only = args.auto_only
symmetric = args.symmetric

#-- Get the input
#- Get the mask and make it a field
start = time.time()
mask = hp.read_map(mask_fits)
fmask = nmt.NmtField(mask, [mask])
print('Get mask took ', time.time()-start, 's', flush=True)

#- Get the Cls
start = time.time()
Cl, keys = get_cosmosis_Cl(ref_cell, nzbins, symmetric)
ncl = len(Cl) # get the number of pairs
if auto_only: ncl = nzbins
print('Get Cl took ', time.time()-start, 's', flush=True)


#- Get the workspace
start = time.time()
workspace = nmt.NmtWorkspace()
workspace.read_from(workspace_fits)
nell = workspace.get_bandpower_windows().shape[1]
print('Get workspace took ', time.time()-start, 's', flush=True)

#-- Initialise covariance workspace
start = time.time()
cov_workspace = nmt.NmtCovarianceWorkspace()
cov_workspace.compute_coupling_coefficients(fmask, fmask)
print('Get cov workspace took ', time.time()-start, 's', flush=True)

#-- Initialise covariance matrix
covmat = np.zeros((ncl*nell, ncl*nell))

#-- Loop over all blocks (pair-pair correlations) to construct the full covariance matrix.
start=time.time()

# Get symetric Cls for permutations
for key in keys:
    probeA, probeB = key.split('-')
    Cl['-'.join([probeB, probeA])] = Cl[key]
    
if auto_only:
    keys = ['-'.join([str(i+1),str(i+1)]) for i in range(nzbins)]

for (idx1, key1), (idx2, key2) in itertools.combinations_with_replacement(enumerate(keys), 2):
    print(key1, key2, flush=True)
    probeA, probeB = key1.split('-')
    probeC, probeD = key2.split('-')
    
    covmat[idx1*nell:(idx1+1)*nell, idx2*nell:(idx2+1)*nell] = nmt.gaussian_covariance(cov_workspace, 0, 0, 0, 0,  # Spins of the 4 fields
                                      [Cl['-'.join([probeA, probeC])]],
                                      [Cl['-'.join([probeB, probeC])]],
                                      [Cl['-'.join([probeA, probeD])]],
                                      [Cl['-'.join([probeB, probeD])]],
                                      workspace, wb=workspace)
covmat = covmat + covmat.T - np.diag(covmat.diagonal())
print('constructing the matrix took ', time.time() - start, 's', flush=True)
#-- Save the covariance matrix
np.save(output_name, covmat)
