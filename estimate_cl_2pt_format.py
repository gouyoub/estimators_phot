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

# #-- Save data to the fits 2PT format
# #- Format vectors of Cl, ell, indices
# ncl = {True  : npairs,
#        False : nzbins}
# # ell
# effective_ell = bnmt.get_effective_ells()
# ell_vec       = np.array( [ effective_ell for i in range(ncl[cross]) ] ).flatten()
# ell_index_vec = np.array( [ np.arange(effective_ell.size) for i in range(ncl[cross]) ] ).flatten()
# # Cl
# Cl_vec        = np.array( [ cl_dic[k] for k in cl_dic.keys() ] ).flatten()
# # indices
# bin1          = np.array([])
# bin2          = np.array([])
# for key in cl_dic.keys():
#     b1   = key.split('-')[0].split(probe)[1]
#     b2   = key.split('-')[1].split(probe)[1]
#     bin1 = np.append(bin1, np.array([b1 for i in effective_ell]))
#     bin2 = np.append(bin2, np.array([b2 for i in effective_ell]))
    
# #- Create the fits file
# hdu = fits.HDUList() # Create hdu list.
# # Append all the vectors to the hdu list element
# fitscol = []
# fitscol.append(fits.Column(name='BIN1', format='K', array=bin1))
# fitscol.append(fits.Column(name='BIN2', format='K', array=bin2))
# fitscol.append(fits.Column(name='ANGBIN', format='K', array=ell_index_vec))
# fitscol.append(fits.Column(name='VALUE', format='D', array=Cl_vec))
# fitscol.append(fits.Column(name='ANG', format='D', array=ell_vec))

# coldefs = fits.ColDefs(fitscol) # Define fits object to be saved
# hdu.append(fits.BinTableHDU.from_columns(coldefs)) # Append to the fits HDU list
# hdu[1].name = 'galaxy_cl' # Give a name to the element

# #- Add all the headers
# hdu[1].header['2PTDATA']   = True
# hdu[1].header['QUANT1']    = 'GPF'
# hdu[1].header['QUANT1']    = 'GPF'
# hdu[1].header['NANGLE']    = effective_ell.size
# hdu[1].header['NBIN_1']    = nzbins
# hdu[1].header['NBIN_2']    = nzbins
# hdu[1].header['WINDOWS']   = 'SAMPLE'
# hdu[1].header['SIMULATED'] = False
# hdu[1].header['BLINDED']   = False
# hdu[1].header['KERNEL_1']  = 'NZ_SOURCE'
# hdu[1].header['KERNEL_2']  = 'NZ_SOURCE'

# #-- Construct the n(z) to save it in the 2PT file
# #- Compute the histogram
# nb = 400 # number of bins for the histogram
# nz = np.zeros((nzbins,nb))
# zb_centers = np.zeros((nzbins,nb))
# zb_min = np.zeros((nzbins,nb))
# zb_max = np.zeros((nzbins,nb))
# for i in range(nzbins):
#     nz[i], b = np.histogram(tomo_bins[i], bins=nb, density=True)
#     zb_centers[i] = np.diff(b)/2.+b[:nb]
#     zb_min[i] = b[:nb]
#     zb_max[i] = b[1:]

# #-- Save the fits file
# hdu.writeto(output_fits+'_'+probe+'_'+redshift_type+'_NZ'+str(nzbins)+'_NS'+str(nside)+'_LMIN'+str(ell_min)+'_BW'+str(binwidth)+'_X'+str(cross)+'.fits', overwrite=True) # Write and save

# # # #### Idea : give as input a file with theory unbinned Cl to be binned in the right way in the script 
# # # #### Idea : compute and output the n(z) as well
