"""
script: estimate_cl

Script to estimate the angular power spectrum for a given catalog
in the CosmoHUB format. For now only the GCph Cl can be estimated.
The outuput should be the one used by CosmoSIS: the 2PT.fits
format.
"""

import numpy as np
import healpy as hp
from collections import OrderedDict

import pymaster as nmt
import anglib as al
from loading import load_it

import sys
import os
import time
import configparser
import itertools as it


config = configparser.ConfigParser()
configname = sys.argv[1]
config.read(configname)
print("-----------------**  ARGUMENTS  **------------------------")
for sec in config.sections():
    print('[{}]'.format(sec))
    for it in config.items(sec):
        print('{} : {}'.format(it[0], it[1]))
    print('')

in_out       = load_it(config._sections['in_out'])
pixels          = load_it(config._sections['pixels'])
probe_selection = load_it(config._sections['probe_selection'])
ell_binning     = load_it(config._sections['ell_binning'])
z_binning       = load_it(config._sections['z_binning'])
noise           = load_it(config._sections['noise'])
apodization     = load_it(config._sections['apodization'])
nside           = pixels['nside']
nz              = len(z_binning['selected_bins'])

#-- Get the mask
mask = hp.read_map(in_out['mask'])
fsky = np.mean(mask)
if apodization['apodize']:
    print('\nApodizing the mask')
    mask = nmt.mask_apodization(mask, apodization['aposcale'],
                                apotype=apodization['apotype'])

#-- Redshift binning
if 'GC' in probe_selection['probes'] or 'GGL' in probe_selection['probes']:
    print(probe_selection['probes'])
    tomo_bins_lens, ngal_bins_lens = al.create_redshift_bins_complete(in_out['catalog_lens'],
                                                z_binning['selected_bins'],
                                                'lens',
                                                z_binning['division'],
                                                z_binning['nofz_redshift_type'],
                                                z_binning['zmin'],
                                                z_binning['zmax'],
                                                z_binning['nztot'])
    ngal_arcmin_lens = (ngal_bins_lens/(4*np.pi*fsky**2))/(((180/np.pi)**2)*3600)

if 'WL' in probe_selection['probes'] or 'GGL' in probe_selection['probes']:
    print(probe_selection['probes'])
    tomo_bins_source, ngal_bins_source = al.create_redshift_bins_complete(in_out['catalog_source'],
                                                z_binning['selected_bins'],
                                                'source',
                                                z_binning['division'],
                                                z_binning['nofz_redshift_type'],
                                                z_binning['zmin'],
                                                z_binning['zmax'],
                                                z_binning['nztot'])
    ngal_arcmin_source = (ngal_bins_source/(4*np.pi*fsky**2))/(((180/np.pi)**2)*3600)



#-- Build n(z)
nofz_name_ref, nofz_name_ext = z_binning['nofz_name'].split('.')
ngal_name_ref, ngal_name_ext = z_binning['ngal_name'].split('.')
nofz_dic = {}
ngal_dic = {}
print('\nSaving the n(z) and galaxy number density')
if 'GC' in probe_selection['probes'] or 'GGL' in probe_selection['probes']:
    nofz_lens = al.build_nz(tomo_bins_lens)
    np.savetxt(nofz_name_ref+'_lens.'+nofz_name_ext, nofz_lens.T)
    np.savetxt(ngal_name_ref+'_lens.'+ngal_name_ext, ngal_arcmin_lens)
    nofz_dic['lens'] = nofz_lens
    ngal_dic['lens'] = ngal_arcmin_lens

if 'WL' in probe_selection['probes'] or 'GGL' in probe_selection['probes']:
    nofz_source = al.build_nz(tomo_bins_source)
    np.savetxt(nofz_name_ref+'_source.'+nofz_name_ext, nofz_source.T)
    np.savetxt(ngal_name_ref+'_source.'+ngal_name_ext, ngal_arcmin_source)
    nofz_dic['source'] = nofz_source
    ngal_dic['source'] = ngal_arcmin_source

if z_binning['only_nofz']:
    print('\nYou only asked for the n(z)')
    print('\nDONE')
    os._exit(0)

#-- Estimate maps
maps_dic = {}
noise_dic = {}
print('\nComputing maps')
compute_map, key_map = al.get_map_for_probes(probe_selection['probes'])
for map, k in zip(compute_map, key_map):
    for i, izb in enumerate(z_binning['selected_bins']):
        print('Map bin{}'.format(izb))
        if k == 'D' :
            tomo_bins = tomo_bins_lens
        if k == 'G' :
            tomo_bins = tomo_bins_source
        maps_dic['{}{}'.format(k,izb+1)] = map(tomo_bins[i], nside, mask)
        noise_dic['{}{}'.format(k,izb+1)] = al.compute_noise(k, tomo_bins[i], fsky)

if pixels['save_maps']:
    map_name = '{}_maps_NS{}'.format(in_out['output_name'],
                                      nside)
    np.save(map_name+'.npy', maps_dic)

#-- Define nmt multipole binning
if ell_binning['ell_binning'] == 'lin':
    bnmt = al.edges_binning(nside, ell_binning['lmin'], ell_binning['binwidth'])

elif ell_binning['ell_binning'] == 'log':
    bnmt = al.edges_log_binning(nside, ell_binning['lmin'], ell_binning['nell'])

#-- Define nmt workspace only with the mask
print('\nGetting the mask and computing the mixing matrix ')
w = nmt.NmtWorkspace()
fmask = nmt.NmtField(mask, [mask], lmax=bnmt.lmax) # nmt field with only the mask
start = time.time()
w.compute_coupling_matrix(fmask, fmask, bnmt) # compute the mixing matrix (which only depends on the mask) just once
w_fname = '{}_NmtWorkspace_NS{}_LBIN{}'.format(in_out['output_name'], nside, ell_binning['ell_binning'])
print('w_fname : ', w_fname)
if ell_binning['ell_binning'] == 'lin':
    w_fname += '_LMIN{}_BW{}'.format(ell_binning['lmin'],
                                     ell_binning['binwidth'])
elif ell_binning['ell_binning'] == 'log':
    w_fname += '_LMIN{}_NELL{}'.format(ell_binning['lmin'],
                                      ell_binning['nell'])
w_fname += '.fits'
w.write_to(w_fname)
print('\n',time.time()-start,'s to compute the coupling matrix')

#- Cl computation loop
cls_dic  = OrderedDict() # To store the cl to be saved in a fit file

for probe in probe_selection['probes']:
    print('\nFor probe {}'.format(probe))
    for pa, pb in al.get_iter(probe, probe_selection['cross'], z_binning['selected_bins']):
        print('Combination is {}-{}'.format(pa,pb))
        fld_a = al.map2fld(maps_dic[pa], mask, bnmt.lmax)
        fld_b = al.map2fld(maps_dic[pb], mask, bnmt.lmax)
        cl = al.compute_master(fld_a, fld_b, w, nside, pixels['depixelate'])
        if pa == pb:
            cl = al.debias(cl, noise_dic[pa], w, nside, noise['debias'], pixels['depixelate'])

        cls_dic['{}-{}'.format(pa,pb)] = cl

cls_dic['ell'] = bnmt.get_effective_ells()

outname = '{}_Cls_NS{}_LBIN{}'.format(in_out['output_name'],
                                      nside,
                                      ell_binning['ell_binning'])

if ell_binning['ell_binning'] == 'lin':
    outname += '_LMIN{}_BW{}'.format(ell_binning['lmin'],
                                     ell_binning['binwidth'])

elif ell_binning['ell_binning'] == 'log':
    outname += '_LMIN{}_NELL{}'.format(ell_binning['lmin'],
                                      ell_binning['nell'])

print('\n Saving to {} format'.format(in_out['output_format']))
if in_out['output_format'] == 'numpy':
    # Save dictionnary to numpy file
    np.save(outname+'.npy', cls_dic)

if in_out['output_format'] == 'twopoint':
    # Save to two point file
    al.save_twopoint(cls_dic,
                     bnmt,
                     nofz_dic,
                     ngal_dic,
                     z_binning['selected_bins'],
                     probe_selection['probes'],
                     probe_selection['cross'],
                     (outname+'.fits'))

print('\nDONE')

