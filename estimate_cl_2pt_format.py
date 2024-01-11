"""
script: estimate_cl

Script to estimate the angular power spectrum for a given catalog
in the CosmoHUB format. For now only the GCph Cl can be estimated.
The outuput should be the one used by CosmoSIS: the 2PT.fits 
format.  
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt
from astropy.io import fits
from scipy import interpolate
from collections import OrderedDict

import time
import argparse

# ------------ FUNCTIONS -----------------------------------------------------------------------------#

#-- Make healpix map of galaxy number for NSIDE from set of angles in RA and Dec
# nbar is computed from all the pixels contained by the mask that was given in input. 
def ang2map_radec(nns, ra, dec):
    #- convert from deg to sr
    start = time.time()
    ra_rad = ra*(np.pi/180); dec_rad = dec*(np.pi/180)
    print(time.time()-start,'s for conversion')
    
    #- get the number of pixels for nside
    npix = hp.nside2npix(nns)
    
    #- get the pixel number for the given angles (need to change dec to theta pi/2.-dec)
    start = time.time()
    pix = hp.pixelfunc.ang2pix(nns, np.pi/2.-dec_rad, ra_rad)
    print(time.time()-start,'s for ang2pix')
    
    #- get the hpmap (i.e. the number of particles per pixels) 
    start = time.time()
    hpmap = np.bincount(pix, weights=np.ones(dec.size), minlength=npix)
    #hpmap, bin_center = np.histogram(pix, np.arange(npix+1))
    print(time.time()-start,'s to create the hpmap')
    
    return hpmap

#-- Make NaMaster fields from hp maps.
# 1) Get the edges of the mask
# 2) Put to zero every pixel of the hp maps outside the mask
# 3) Compute nbar from all the pixels inside the mask and Compute delta = n/nbar - 1
# 4) Create a field from this map
def map2fld(hpmap1, hpmap2, mask):
    #- Get the nside from npix (which is the size of the map)
    nns = hp.npix2nside(hpmap1.size)
        
    start = time.time()
    #- Get coordinates of all pixels
    thetapix, phipix = hp.pix2ang(nns, np.arange(hpmap1.size))
    print(time.time()-start,'s for pix2ang')

    #- Get max and min theta and phi of the mask
    start = time.time()
    tmi = thetapix[mask == 1 ].min()
    tma = thetapix[mask == 1 ].max()
    pmi = phipix[mask == 1 ].min()
    pma = phipix[mask == 1 ].max()
    print(time.time()-start,'s for min and max mask')
    
    #- Define conditions in and out of the mask
    start = time.time()    
    cond_out_mask = (thetapix<tmi) | (thetapix>tma) | (phipix<pmi) | (phipix>pma)
    cond_in_mask = (thetapix>=tmi) & (thetapix<=tma) & (phipix>=pmi) & (phipix<=pma)
    print(time.time()-start,'s to define selection')   
    
    #- Cut the input HP map to the mask
    start = time.time()
    hpmap1[cond_out_mask] = 0.
    hpmap2[cond_out_mask] = 0.
    print(time.time()-start,'s for cut')
    
    #- Compute shotnoise
    sn = (4*np.pi*compute_fsky(mask)**2)/hpmap1.sum()
    
    #- Compute nbar from all the pixels inside the mask and compute delta
    start = time.time()
    nbar1 = np.mean(hpmap1[cond_in_mask])
    nbar2 = np.mean(hpmap2[cond_in_mask])
    hpmap1[cond_in_mask] = hpmap1[cond_in_mask]/nbar1 - 1
    hpmap2[cond_in_mask] = hpmap2[cond_in_mask]/nbar2 - 1
    print(time.time()-start,'s for nbar')

    #- Create NaMaster field
    start = time.time()
    fld1 = nmt.NmtField(mask, [hpmap1])
    fld2 = nmt.NmtField(mask, [hpmap2])
    print('Mean density contrast : ', hpmap1[cond_in_mask].mean())
    print(time.time()-start,'s for field')
    
    return fld1, fld2, sn

#-- Estimate Cl from a pair of field and for a workspace that contains the pre-computed mixing matrix
def compute_master(f_a, f_b, wsp, nns):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)/hp.sphtfunc.pixwin(nns)**2
    cl_decoupled = wsp.decouple_cell(cl_coupled)

    return cl_decoupled

#-- Perform the whole chain of processes to estimate Cl from a pair of (ra, dec) set, for a given NaMaster workspace that 
# contains the mask, the binning and the coupling matrix.
#- 1) Project the tomographic bin of galaxy positions to a hp map. In other word get the hpmap from (ra, dec)
#- 2) Get the associated galaxy overdensity fields f_1 and f_2 in a NaMaster format
#- 3) Estimate Cl of f_1 X f_2
def ang2cl(nns, tomobin1, tomobin2, mask, wsp):
    hpmap1 = ang2map_radec(nns, tomobin1['ra'], tomobin1['dec'])
    hpmap2 = ang2map_radec(nns, tomobin2['ra'], tomobin2['dec'])
    
    f1, f2, sn = map2fld(hpmap1, hpmap2, mask)
    
    start = time.time()
    cl = compute_master(f1, f2, wsp, nns)
    print(time.time()-start,'s for Cl estimation')
    
    #- decouple the shotnoise
    snl = np.array([ np.full(3 * nns, sn) ])/hp.sphtfunc.pixwin(nns)**2
    snl = wsp.decouple_cell(snl)
    
    return cl, snl

#-- Define a ell binning from NSIDE and bw for a given lmin
def edges_binning(NSIDE, lmin, bw):
    lmax = 2*NSIDE
    nbl = (lmax-lmin)//bw + 1
    elli = np.zeros(nbl, int)
    elle = np.zeros(nbl, int)

    for i in range(nbl):
        elli[i] = lmin + i*bw
        elle[i] = lmin + (i+1)*bw

    b = nmt.NmtBin.from_edges(elli, elle)
    return b

def compute_fsky(mask):
    i_zeros = np.where(mask != 0)
    return float(i_zeros[0].size)/float(mask.size)

# --------------------------------------------------------------------------------------------------#

#-- Argument parser
parser = argparse.ArgumentParser(description='Script to estimate angular power spectrum with NaMaster (for now just GC but will also do WL and GGL). Input data has to be in a fits file')
parser.add_argument('-i', '--input', dest='input_fits', type=str, required=True, help='Absolute path to input fits file. Example : /nfs/pic.es/user/s/sgouyoub/euclid/flagship/data/FS2_galaxies_obsz_photoz_ra_dec_mag_magcut245_fullsample.fits')
parser.add_argument('-o', '--output', dest='output_fits', type=str, required=True, help='Absolute path to the output fits file containing the estimated Cl. The name of the file will be the ref given at the end + _probes_nzbins_nside_ellbin.fits . Example : /nfs/pic.es/user/s/sgouyoub/euclid/flagship/data/blabla_GC_NZ13_NS256_BW5.fits')
parser.add_argument('-m', '--mask', dest='mask_fits', type=str, required=True, help='Absolute path to the fits file containing the mask. It has to has been generated with the same NSIDE used here. Example : /nfs/pic.es/user/s/sgouyoub/euclid/flagship/data/flagship2_mask_binary_NS256.fits')
parser.add_argument('-p', '--probe', dest='probe', type=str, required=True, help='The statistics to estimate.', choices=['GC'])
parser.add_argument('-z', '--ztype', dest='redshift_type', type=str, required=True, help='The type of redshift to consider', choices=['zp', 'true_redshift_gal', 'observed_redshift_gal'])
parser.add_argument('-ns', '--nside', dest='nside', type=int, required=True, help='The nside of healpix maps')
parser.add_argument('-l', '--lmin', dest='ell_min', type=int, required=True, help='The minimum ell')
parser.add_argument('-nz', '--nzbins', dest='nzbins', type=int, required=True, help='The number of galaxy equidistant tomographic bins')
parser.add_argument('-zmi', '--zmin', dest='zmin', type=float, required=True, help='Minimum redshift for the redshift binning')
parser.add_argument('-zma', '--zmax', dest='zmax', type=float, required=True, help='Maximum redshift for the redshift binning')
parser.add_argument('-bw', '--binwidth', dest='binwidth', type=int, required=True, help='The width of the multipole bins')
parser.add_argument('-x', dest='cross', action='store_true', help='Compute also the cross-Cl. Else, only compute and save the auto-Cl.')
args = parser.parse_args()
dictargs = vars(args)

print("-----------------**  ARGUMENTS  **------------------------")
for keys in dictargs.keys():
    print(keys, " = ", dictargs[keys])

#-- Arguments
input_fits = args.input_fits
output_fits = args.output_fits
mask_fits = args.mask_fits
probe = args.probe
redshift_type = args.redshift_type
nside = args.nside
ell_min = args.ell_min
nzbins = args.nzbins
zmin = args.zmin
zmax = args.zmax
binwidth = args.binwidth
cross = args.cross


#-- Get the mask
mask = hp.read_map(mask_fits)
fsky = compute_fsky(mask)

#-- Load the catalog
hdul = fits.open(input_fits)
print(hdul[1].header)

#-- Redshift binning
#- Edges of the bins
z_edges = np.linspace(zmin, zmax, nzbins+1)
print("z_edges : ", z_edges)

#- Selection
tomo_bins=[]
shotnoise=np.zeros(nzbins)
for i in range(nzbins):
    selection = (hdul[1].data[redshift_type]>=z_edges[i]) & (hdul[1].data[redshift_type]<=z_edges[i+1])
    tbin = {}
    tbin['ra'] = hdul[1].data['ra_gal'][selection]
    tbin['dec'] = hdul[1].data['dec_gal'][selection]
    tbin['ngal'] = tbin['dec'].size
    tbin['nofz'] = hdul[1].data['observed_redshift_gal'][selection]
    tomo_bins.append(tbin)
hdul.close()

#-- Estimate all auto- and cross-Cl if asked
#- Define nmt multipole binning
#bnmt = nmt.NmtBin.from_nside_linear(nside, binwidth)
bnmt = edges_binning(nside, ell_min, binwidth)

#- Define nmt workspace only with the mask
w = nmt.NmtWorkspace()
fmask = nmt.NmtField(mask, [mask]) # nmt field with only the mask
start = time.time()
w.compute_coupling_matrix(fmask, fmask, bnmt) # compute the mixing matrix (which only depends on the mask) just once
w.write_to(output_fits+'_NmtWorkspace_NS'+str(nside)+'_LMIN'+str(ell_min)+'_BW'+str(binwidth)+'.fits')
print('\n',time.time()-start,'s to compute the coupling matrix')

#- Cl computation loop
cl_dic  = OrderedDict() # To store the cl to be saved in a fit file
snl_dic = OrderedDict() # To store the shotnoise to be saved in a fit file
npairs  = None
if cross:
    npairs = nzbins*(nzbins-1)/2 + nzbins
    print('There is a total of ', str(npairs), ' auto- and cross-Cl to estimate')
    for i in range(nzbins):
        for j in range(nzbins):
            print('\n Estimating for pair ij = ',str(i+1)+'-'+str(j+1) )
            result, shotnoise = ang2cl(nside, tomo_bins[i], tomo_bins[j], mask, w)
            cl_dic[probe+str(i+1)+'-'+probe+str(j+1)] = result[0]
            #fitscol_cl.append(fits.Column(name=probe+str(i+1)+str(j+1), format='D', array=result[0])) # Store Cl
        snl_dic['SN'+str(i+1)] = shotnoise[0]
        #fitscol_snl.append(fits.Column(name='shotnoise'+str(i+1), format='D', array=shotnoise[0])) # Store SN
        

else :
    print('There is a total of ', str(nzbins), ' auto-Cl to estimate')
    for i in range(nzbins):
        print('\n Estimating for pair ij = ',str(i+1)+'-'+str(i+1) )
        result, shotnoise  = ang2cl(nside, tomo_bins[i], tomo_bins[i], mask, w)
        cl_dic[probe+str(i+1)+'-'+probe+str(i+1)] = result[0]
        #fitscol_cl.append(fits.Column(name=probe+str(i+1)+str(i+1), format='D', array=result[0])) # Store Cl
        snl_dic['SN'+str(i+1)] = shotnoise[0]
        #fitscol_snl.append(fits.Column(name='shotnoise'+str(i+1), format='D', array=shotnoise[0])) # Store SN

#-- Save data to the fits 2PT format
#- Format vectors of Cl, ell, indices
ncl = {True  : npairs,
       False : nzbins}
# ell
effective_ell = bnmt.get_effective_ells()
ell_vec       = np.array( [ effective_ell for i in range(ncl[cross]) ] ).flatten()
ell_index_vec = np.array( [ np.arange(effective_ell.size) for i in range(ncl[cross]) ] ).flatten()
# Cl
Cl_vec        = np.array( [ cl_dic[k] for k in cl_dic.keys() ] ).flatten()
# indices
bin1          = np.array([])
bin2          = np.array([])
for key in cl_dic.keys():
    b1   = key.split('-')[0].split(probe)[1]
    b2   = key.split('-')[1].split(probe)[1]
    bin1 = np.append(bin1, np.array([b1 for i in effective_ell]))
    bin2 = np.append(bin2, np.array([b2 for i in effective_ell]))
    
#- Create the fits file
hdu = fits.HDUList() # Create hdu list.
# Append all the vectors to the hdu list element
fitscol = []
fitscol.append(fits.Column(name='BIN1', format='K', array=bin1))
fitscol.append(fits.Column(name='BIN2', format='K', array=bin2))
fitscol.append(fits.Column(name='ANGBIN', format='K', array=ell_index_vec))
fitscol.append(fits.Column(name='VALUE', format='D', array=Cl_vec))
fitscol.append(fits.Column(name='ANG', format='D', array=ell_vec))

coldefs = fits.ColDefs(fitscol) # Define fits object to be saved
hdu.append(fits.BinTableHDU.from_columns(coldefs)) # Append to the fits HDU list
hdu[1].name = 'galaxy_cl' # Give a name to the element

#- Add all the headers
hdu[1].header['2PTDATA']   = True
hdu[1].header['QUANT1']    = 'GPF'
hdu[1].header['QUANT1']    = 'GPF'
hdu[1].header['NANGLE']    = effective_ell.size
hdu[1].header['NBIN_1']    = nzbins
hdu[1].header['NBIN_2']    = nzbins
hdu[1].header['WINDOWS']   = 'SAMPLE'
hdu[1].header['SIMULATED'] = False
hdu[1].header['BLINDED']   = False
hdu[1].header['KERNEL_1']  = 'NZ_SOURCE'
hdu[1].header['KERNEL_2']  = 'NZ_SOURCE'

#-- Construct the n(z) to save it in the 2PT file
#- Compute the histogram
nb = 400 # number of bins for the histogram
nz = np.zeros((nzbins,nb))
zb_centers = np.zeros((nzbins,nb))
zb_min = np.zeros((nzbins,nb))
zb_max = np.zeros((nzbins,nb))
for i in range(nzbins):
    nz[i], b = np.histogram(tomo_bins[i], bins=nb, density=True)
    zb_centers[i] = np.diff(b)/2.+b[:nb]
    zb_min[i] = b[:nb]
    zb_max[i] = b[1:]

#-- Save the fits file
hdu.writeto(output_fits+'_'+probe+'_'+redshift_type+'_NZ'+str(nzbins)+'_NS'+str(nside)+'_LMIN'+str(ell_min)+'_BW'+str(binwidth)+'_X'+str(cross)+'.fits', overwrite=True) # Write and save

# #### Idea : give as input a file with theory unbinned Cl to be binned in the right way in the script 
# #### Idea : compute and output the n(z) as well
