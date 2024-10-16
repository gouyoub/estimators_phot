import numpy as np
import pymaster as nmt
import healpy as hp

import loading

import sys
import itertools
import time
import configparser

config = configparser.ConfigParser()
configname = sys.argv[1]
config.read(configname)
print("-----------------**  ARGUMENTS  **------------------------")
for sec in config.sections():
    print('[{}]'.format(sec))
    for it in config.items(sec):
        print('{} : {}'.format(it[0], it[1]))
    print('')

# -- Arguments
filenames = loading.load_it(config._sections['filenames'])
probe_selection = loading.load_it(config._sections['probe_selection'])
noise = loading.load_it(config._sections['noise'])
mask_fits = filenames['mask']
ref_cell = filenames['cell']
workspace_fits = filenames['workspace']
output_name = filenames['output']
nzbins = probe_selection['nzbins']
cross = probe_selection['cross']
probe = probe_selection['probe']
add_noise = noise['add_noise']
ng_shear_file = noise['ng_shear_arcmin']
ng_density_file = noise['ng_density_arcmin']
sigma_e_tot = noise['sigma_e_tot']

# -- Get the input
# - Get the mask and make it a field
start = time.time()
mask = hp.read_map(mask_fits)
fmask = nmt.NmtField(mask, [mask])
print('Get mask took ', time.time()-start, 's', flush=True)

# - Get the Cls
start = time.time()
Cl = {}
keys = []
keys_all = []

if len(probe)>1 or probe == ['GGL']:
    for p in ['GC', 'GGL', 'WL']:
        Cl_temp, keys_all_temp = loading.get_cosmosis_Cl(ref_cell, nzbins, p, True)
        Cl.update(Cl_temp)
        keys_all += keys_all_temp
    for (p,c) in zip(probe, cross):
        _, keys_temp = loading.get_cosmosis_Cl(ref_cell, nzbins, p, c)
        keys += keys_temp
else:
    Cl_temp, keys_all_temp = loading.get_cosmosis_Cl(ref_cell, nzbins, probe[0], True)
    Cl.update(Cl_temp)
    keys_all += keys_all_temp
    _, keys_temp = loading.get_cosmosis_Cl(ref_cell, nzbins, probe[0], cross[0])
    keys += keys_temp

ncl = len(keys)  # get the number of pairs
print('The number of Cls is : ', ncl)
print('Get Cl took ', time.time()-start, 's', flush=True)

# - Get galaxy number density
ng_density = np.loadtxt(ng_density_file)
ng_shear = np.loadtxt(ng_shear_file)

# - Get the workspace
start = time.time()
workspace = nmt.NmtWorkspace()
workspace.read_from(workspace_fits)
nell = workspace.get_bandpower_windows().shape[1]
print('Get workspace took ', time.time()-start, 's', flush=True)

# Check shape of Cl's for given workspace

assert Cl[keys[0]].size >= workspace.wsp.lmax_mask+1, 'Cls have wrong lenght. It should be at least lmax*3+1'

# -- Add noise term
arcmin2deg = ((180/np.pi)**2)*3600
if add_noise:
    for p in [p for p in probe if p != "GGL"]:
        _, keys_auto = loading.get_cosmosis_Cl(ref_cell, nzbins, p, False)
        for zi,k in enumerate(keys_auto):
            noise_term = 1/(ng_density[zi]*arcmin2deg)
            if p == 'WL': noise_term *= (sigma_e_tot[zi]**2)/2.
            Cl[k] += noise_term

# -- Initialise covariance workspace
start = time.time()
cov_workspace = nmt.NmtCovarianceWorkspace()
cov_workspace.compute_coupling_coefficients(fmask, fmask)
print('Get cov workspace took ', time.time()-start, 's', flush=True)

# -- Initialise covariance matrix
covmat = np.zeros((ncl*nell, ncl*nell))

# -- Loop over all blocks (pair-pair correlations) to construct the full covariance matrix.
start = time.time()

# Get symetric Cls for permutations
for p in probe:
    for key in keys_all:
        probeA, probeB = key.split('-')
        Cl['-'.join([probeB, probeA])] = Cl[key]

# if cross == False:
#     keys = ['-'.join([str(i+1), str(i+1)]) for i in range(nzbins)]

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
