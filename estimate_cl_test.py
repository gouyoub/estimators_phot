# """
# script: estimate_cl

# Script to estimate the angular power spectrum for a given catalog
# in the CosmoHUB format. For now only the GCph Cl can be estimated.
# The outuput should be the one used by CosmoSIS: the 2PT.fits 
# format.  
# """
import numpy as np
import healpy as hp
from astropy.io import fits
from collections import OrderedDict

import pymaster as nmt
import anglib as al
import save
from loading import load_it

import time
import configparser
import itertools as it

config = configparser.ConfigParser()
config.read('inifiles/example.cfg')
print("-----------------**  ARGUMENTS  **------------------------")
for sec in config.sections():
    print('[{}]'.format(sec))
    for it in config.items(sec):
        print('{} : {}'.format(it[0], it[1]))
    print('')

filenames       = load_it(config._sections['filenames'])
pixels          = load_it(config._sections['pixels'])
probe_selection = load_it(config._sections['probe_selection'])
ell_binning     = load_it(config._sections['ell_binning'])
z_binning       = load_it(config._sections['z_binning'])
noise           = load_it(config._sections['noise'])
nside           = pixels['nside']
nz              = len(z_binning['selected_bins'])

#-- Get the mask
mask = hp.read_map(filenames['mask'])

#-- Redshift binning
tomo_bins, ngal_bins = al.create_redshift_bins(filenames['catalog'], 
                                               z_binning['selected_bins'], 
                                               z_binning['zmin'], 
                                               z_binning['zmax'],
                                               z_binning['nztot'])
    
#-- Estimate maps
maps_dic = {}
noise_dic = {}
compute_map, key_map = al.get_map_for_probes(probe_selection['probes'])
for map, k in zip(compute_map, key_map):
    for i, izb in enumerate(z_binning['selected_bins']):
        print('Map bin{}'.format(izb))
        maps_dic['{}{}'.format(k,izb)], noise_dic['{}{}'.format(k,izb)] = map(tomo_bins[i], nside, mask)
#-- Define nmt multipole binning
bnmt = al.edges_binning(nside, ell_binning['lmin'], ell_binning['binwidth'])

#-- Define nmt workspace only with the mask
w = nmt.NmtWorkspace()
fmask = nmt.NmtField(mask, [mask]) # nmt field with only the mask
start = time.time()
w.compute_coupling_matrix(fmask, fmask, bnmt) # compute the mixing matrix (which only depends on the mask) just once
w_fname = '{}_NmtWorkspace_NS{}_LMIN{}_BW{}.fits'.format(filenames['output'], nside, ell_binning['lmin'], ell_binning['binwidth'])
w.write_to(w_fname)    
print('\n',time.time()-start,'s to compute the coupling matrix')

#- Cl computation loop
cls_dic  = OrderedDict() # To store the cl to be saved in a fit file
for probe in probe_selection['probes']:
    print('\nFor probe {}'.format(probe))
    for pa, pb in al.get_iter(probe, probe_selection['cross'], z_binning['selected_bins']):
        print('Combination is {}-{}'.format(pa,pb))
        fld_a = al.map2fld(maps_dic[pa], mask)
        fld_b = al.map2fld(maps_dic[pb], mask)
        
        cl = al.compute_master(fld_a, fld_b, w, nside, pixels['depixelate'])
        if noise['debias'] and (pa == pb):
            N = al.decouple_noise(noise_dic[pa], w, nside, pixels['depixelate'])
            cl -= N
            
        cls_dic['{}-{}'.format(pa,pb)] = cl
              
cls_dic['ell'] = bnmt.get_effective_ells()
      
#- Save dictionnary to file
outname = '{}_NS{}_BW{}_LMIN{}_depix{}_debias{}'.format(filenames['output'],
                                                             nside,
                                                             ell_binning['binwidth'],
                                                             ell_binning['lmin'],
                                                             pixels['depixelate'],
                                                             noise['debias'])
save.numpy_save(outname, cls_dic)
print('DONE')
