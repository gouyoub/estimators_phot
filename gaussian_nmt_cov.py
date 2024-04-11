import numpy as np
import pymaster as nmt
import healpy as hp

import loading

import itertools
import time
import configparser

config = configparser.ConfigParser()
config.read('inifiles/example_cov.cfg')
print("-----------------**  ARGUMENTS  **------------------------")
for sec in config.sections():
    print('[{}]'.format(sec))
    for it in config.items(sec):
        print('{} : {}'.format(it[0], it[1]))
    print('')

# -- Arguments
filenames = loading.load_it(config._sections['filenames'])
probe_selection = loading.load_it(config._sections['probe_selection'])
mask_fits = filenames['mask']
ref_cell = filenames['cell']
workspace_fits = filenames['workspace']
output_name = filenames['output']
nzbins = probe_selection['nzbins']
cross = probe_selection['cross']
probe = probe_selection['probe']

# -- Get the input
# - Get the mask and make it a field
start = time.time()
mask = hp.read_map(mask_fits)
fmask = nmt.NmtField(mask, [mask])
print('Get mask took ', time.time()-start, 's', flush=True)

# - Get the Cls
start = time.time()
Cl, keys = loading.get_cosmosis_Cl(ref_cell, nzbins, probe, cross)
# keys = Cl.keys()
ncl = len(Cl)-1  # get the number of pairs
if cross == False:
    ncl = nzbins
print('Get Cl took ', time.time()-start, 's', flush=True)

# - Get the workspace
start = time.time()
workspace = nmt.NmtWorkspace()
workspace.read_from(workspace_fits)
nell = workspace.get_bandpower_windows().shape[1]
print('Get workspace took ', time.time()-start, 's', flush=True)

# -- Initialise covariance workspace
start = time.time()
cov_workspace = nmt.NmtCovarianceWorkspace()
cov_workspace.compute_coupling_coefficients(fmask, fmask)
print('Get cov workspace took ', time.time()-start, 's', flush=True)

# -- Initialise covariance matrix
covmat = np.zeros((ncl*nell, ncl*nell))
print(ncl)

# -- Loop over all blocks (pair-pair correlations) to construct the full covariance matrix.
start = time.time()

# Get symetric Cls for permutations
for key in keys:
    probeA, probeB = key.split('-')
    Cl['-'.join([probeB, probeA])] = Cl[key]

if cross == False:
    keys = ['-'.join([str(i+1), str(i+1)]) for i in range(nzbins)]

for (idx1, key1), (idx2, key2) in itertools.combinations_with_replacement(enumerate(keys), 2):
    print(key1, key2, flush=True)
    probeA, probeB = key1.split('-')
    probeC, probeD = key2.split('-')

    covmat[idx1*nell:(idx1+1)*nell, idx2*nell:(idx2+1)*nell] =\
        nmt.gaussian_covariance(cov_workspace, 0, 0, 0, 0,
                                [Cl['-'.join([probeA, probeC])]],
                                [Cl['-'.join([probeB, probeC])]],
                                [Cl['-'.join([probeA, probeD])]],
                                [Cl['-'.join([probeB, probeD])]],
                                workspace, wb=workspace)

covmat = covmat + covmat.T - np.diag(covmat.diagonal())
print('constructing the matrix took ', time.time() - start, 's', flush=True)
# -- Save the covariance matrix
np.save(output_name, covmat)
