"""
script: estimate_cl

Author: Sylvain Gouyou Beauchamps

Script to estimate the angular power spectrum for a given catalog of source
and lenses using NaMaster.

The script is executed by passing the configuration file as the first argument.

Example:
   python script_name.py config_file.ini

Output:
- Angular power spectra (Cl’s) stored in either Numpy or FITS format.
- Computed maps, noise models, and galaxy distributions stored in Numpy files..

"""

import numpy as np
import healpy as hp
from collections import OrderedDict

import pymaster as nmt
import anglib as al
from general import load_it

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
    print(f"[{sec}]")
    for it in config.items(sec):
        print(f"{it[0]} : {it[1]}")
    print('')

in_out       = load_it(config._sections['in_out'])
maps          = load_it(config._sections['maps'])
probe_selection = load_it(config._sections['probe_selection'])
ell_binning     = load_it(config._sections['ell_binning'])
z_binning       = load_it(config._sections['z_binning'])
noise           = load_it(config._sections['noise'])
apodization     = load_it(config._sections['apodization'])
spectra         = load_it(config._sections['spectra'])
columns_lens    = load_it(config._sections['columns_lens'])
columns_source    = load_it(config._sections['columns_source'])
nside           = maps['nside']
nz              = len(z_binning['selected_bins'])


#-- Checks
if maps['load_maps']:
    assert maps['save_maps'] is False, 'Maps are not saved if they are loaded.'

if ell_binning['lmax'] > 3*nside:
    raise ValueError(f"The ell max is larger than than 3 x NSIDE !")

if os.path.isfile(in_out['output_dir']):
    raise FileExistsError(f"This output directory already exists !")

#-- Managing filenames
print("\nCreating the output directory")
os.mkdir(in_out['output_dir'])
os.system(f"cp {configname} {in_out['output_dir']}") # copy inifile in the output directory
ref_out_fname = f"{in_out['output_dir']}/{in_out['output_dir'].split('/')[-1]}"

#-- Get the mask
mask = hp.read_map(in_out['mask'])
fsky = np.mean(mask)
if apodization['apodize']:
    print('\nApodizing the mask')
    mask = nmt.mask_apodization(mask, apodization['aposcale'],
                                apotype=apodization['apotype'])
# Check consistency of nside
if nside != hp.npix2nside(mask.size):
    raise ValueError(f"The input nside and the mask are not consistent !")

#-- Tomographic binning + Generate maps if not directly loaded
if not maps['load_maps']:
    #-- Redshift binning
    if 'GC' in probe_selection['probes'] or 'GGL' in probe_selection['probes']:
        print(probe_selection['probes'])
        tomo_bins_lens, ngal_bins_lens, z_edges_lens = al.create_redshift_bins(in_out['catalog_lens'],
                                                    columns_lens,
                                                    z_binning['selected_bins'],
                                                    'lens',
                                                    z_binning['division'],
                                                    z_binning['nofz_redshift_type'],
                                                    z_binning['zmin'],
                                                    z_binning['zmax'],
                                                    z_binning['nztot'])
        ngal_arcmin_lens = (ngal_bins_lens/(4*np.pi*fsky))/(((180/np.pi)**2)*3600)

    if 'WL' in probe_selection['probes'] or 'GGL' in probe_selection['probes']:
        print(probe_selection['probes'])
        tomo_bins_source, ngal_bins_source, z_edges_source = al.create_redshift_bins(in_out['catalog_source'],
                                                    columns_source,
                                                    z_binning['selected_bins'],
                                                    'source',
                                                    z_binning['division'],
                                                    z_binning['nofz_redshift_type'],
                                                    z_binning['zmin'],
                                                    z_binning['zmax'],
                                                    z_binning['nztot'])
        ngal_arcmin_source = (ngal_bins_source/(4*np.pi*fsky))/(((180/np.pi)**2)*3600)

    #-- Build n(z)
    nofz_out_fname = f"{ref_out_fname}_nofz"
    ngal_out_fname = f"{ref_out_fname}_ngal"
    z_edges_out_fname = f"{ref_out_fname}_zedges"
    nofz_dic = {}
    ngal_dic = {}
    print('\nSaving the n(z), galaxy number density and z_edges')
    if 'GC' in probe_selection['probes'] or 'GGL' in probe_selection['probes']:
        nofz_lens = al.build_nz(tomo_bins_lens)
        np.savetxt(nofz_out_fname+'_lens.txt', nofz_lens.T)
        np.savetxt(ngal_out_fname+'_arcmin2_lens.txt', ngal_arcmin_lens)
        np.savetxt(z_edges_out_fname+'_lens.txt', z_edges_lens)

        nofz_dic['lens'] = nofz_lens
        ngal_dic['lens'] = ngal_arcmin_lens

    if 'WL' in probe_selection['probes'] or 'GGL' in probe_selection['probes']:
        nofz_source = al.build_nz(tomo_bins_source)
        np.savetxt(nofz_out_fname+'_source.txt', nofz_source.T)
        np.savetxt(ngal_out_fname+'_arcmin2_source.txt', ngal_arcmin_source)
        np.savetxt(z_edges_out_fname+'_source.txt', z_edges_source)
        nofz_dic['source'] = nofz_source
        ngal_dic['source'] = ngal_arcmin_source

    np.save(nofz_out_fname+'.npy', nofz_dic)
    np.save(ngal_out_fname+'.npy', ngal_dic)

    #-- Estimate maps
    maps_dic = {}
    noise_dic = {}
    var_shape = np.zeros(nz)
    print('\nComputing maps')
    compute_map, key_map = al.get_map_for_probes(probe_selection['probes'])
    for map, k in zip(compute_map, key_map):
        for i, izb in enumerate(z_binning['selected_bins']):
            print(f"Map bin{izb}")
            if k == 'D' :
                tomo_bins = tomo_bins_lens
            if k == 'G' :
                tomo_bins = tomo_bins_source
                var_shape[i] = al.shape_var(tomo_bins[i])

            maps_dic[f"{k}{izb+1}"] = map(tomo_bins[i], nside, mask)
            noise_dic[f"{k}{izb+1}"] = al.compute_noise(k, tomo_bins[i], fsky)
    np.savetxt(f"{ref_out_fname}_shapevar.txt", var_shape)

#-- Load maps, nofz, noise if asked
else :
    print("\nThe maps, noise, nofz and ngal will be loaded from external files")
    maps_noise_dic = np.load(in_out['maps_noise_name'], allow_pickle=True).item()
    maps_dic = maps_noise_dic['maps']
    noise_dic = maps_noise_dic['noise']

    nofz_dic = np.load(in_out['nofz_name'], allow_pickle=True).item()
    ngal_dic = np.load(in_out['ngal_name'], allow_pickle=True).item()

#-- Save maps and associated noise if asked
if maps['save_maps']:
    print("\nSaving the maps and noise")
    maps_noise_out_fname = f"{ref_out_fname}_maps_noise_NS{nside}.npy"
    maps_noise_dic = {'maps'  : maps_dic, 'noise' : noise_dic}
    np.save(maps_noise_out_fname, maps_noise_dic)

#-- Define nmt multipole binning
if ell_binning['ell_binning'] == 'lin':
    bnmt = al.linear_binning(ell_binning['lmax'], ell_binning['lmin'], ell_binning['binwidth'])

elif ell_binning['ell_binning'] == 'log':
    bnmt = al.log_binning(ell_binning['lmax'], ell_binning['lmin'], ell_binning['nell'])

else:
    raise ValueError(f"ell_binning must be 'lin' or 'log'")

#-- Define nmt workspace only with the mask
print('\nGetting the mask and computing the mixing matrices ')
start = time.time()

w_fname_base = f"{ref_out_fname}_NmtWorkspace_NS{nside}_LBIN{ell_binning['ell_binning']}"

if ell_binning['ell_binning'] == 'lin':
    w_fname_base += f"_LMIN{ell_binning['lmin']}_LMAX{ell_binning['lmax']}_BW{ell_binning['binwidth']}"

elif ell_binning['ell_binning'] == 'log':
    w_fname_base += f"_LMIN{ell_binning['lmin']}_LMAX{ell_binning['lmax']}_NELL{ell_binning['nell']}"

w_dic = {}
w_arr_dic = {}
for probe in probe_selection['probes']:
    w_fname = f"{w_fname_base}_{al.probe_ref_mapping(probe)}.fits"
    w_dic[probe] = al.create_workspaces(bnmt, mask, w_fname, probe)
    w_arr_dic[probe] =  w_dic[probe].get_coupling_matrix() #Namaster WS to array
    print('w_fname : ', w_fname)

print('\n',time.time()-start,'s to compute the coupling matrices')

#-- Cl computation loop
cls_dic  = OrderedDict() # To store the cl to be saved in a fits file
for probe in probe_selection['probes']:
    print(f"\nFor probe {probe}")
    for pa, pb in al.get_iter(probe, probe_selection['cross'], z_binning['selected_bins']):
        print(f"Combination is {pa}-{pb}")

        # Define fields to be correlated
        fld_a = al.map2fld(maps_dic[pa], mask, bnmt.lmax)
        fld_b = al.map2fld(maps_dic[pb], mask, bnmt.lmax)

        # Compute Peudo-Cl's or bandpowers
        if spectra['decoupling']:
            cl = al.compute_master(fld_a, fld_b, w_dic[probe], nside, maps['depixelate'])
        else:
            cl = al.compute_coupled(fld_a, fld_b, bnmt, nside, maps['depixelate'])

        # Remove noise bias from auto correlation if asked
        if pa == pb:
            cl = al.debias(cl, noise_dic[pa], w_dic[probe], bnmt, fsky, nside,
                           noise['debias'], maps['depixelate'], spectra['decoupling'])

        cls_dic[f"{pa}-{pb}"] = cl

cls_dic['ell'] = bnmt.get_effective_ells()

#-- Saving Cl's
outname = f"{ref_out_fname}_Cls_NS{nside}_LBIN{ell_binning['ell_binning']}"

if ell_binning['ell_binning'] == 'lin':
    outname += f"_LMIN{ell_binning['lmin']}_LMAX{ell_binning['lmax']}_BW{ell_binning['binwidth']}"

elif ell_binning['ell_binning'] == 'log':
    outname += f"_LMIN{ell_binning['lmin']}_LMAX{ell_binning['lmax']}_NELL{ell_binning['nell']}"

print(f"\n Saving to {in_out['output_format']} format")
if in_out['output_format'] == 'numpy':
    # Save dictionnary to numpy file
    np.save(outname+".npy", cls_dic)

elif in_out['output_format'] == 'twopoint':
    # Save to two point file
    al.save_twopoint(cls_dic,
                     bnmt,
                     nofz_dic,
                     ngal_dic,
                     z_binning['selected_bins'],
                     probe_selection['probes'],
                     probe_selection['cross'],
                     (outname+".fits"))
    
elif in_out['output_format'] == 'euclidlib':
    al.save_euclidlib(cls_dic,
                      w_arr_dic,
                      bnmt,
                      z_binning['selected_bins'],
                      probe_selection['probes'],
                      probe_selection['cross'],
                      (outname+"_elib.fits"))

else :
    raise ValueError(f"The format for the C(l) output file should be numpy or twopoint")


print('\nDONE')

