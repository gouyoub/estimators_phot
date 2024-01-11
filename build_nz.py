import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate

import argparse

#-- Argument parser
parser = argparse.ArgumentParser(description='Script to build the n(z) for a photometric sample. Input data has to be in a fits file')
parser.add_argument('-i', '--input', dest='input_fits', type=str, required=True, help='Absolute path to input fits file. Example : /nfs/pic.es/user/s/sgouyoub/euclid/flagship/data/FS2_galaxies_obsz_photoz_ra_dec_mag_magcut245_fullsample.fits')
parser.add_argument('-o', '--output', dest='output_fits', type=str, required=True, help='Absolute path to the output fits file containing the estimated Cl. The name of the file will be the ref given at the end + _probes_nzbins_nside_ellbin.fits . Example : /nfs/pic.es/user/s/sgouyoub/euclid/flagship/data/blabla_NZ13.fits')
parser.add_argument('-zs', '--zselec', dest='zselec', type=str, required=True, help='Redshifts to use to make the selection', choices=['zp', 'true_redshift_gal', 'observed_redshift_gal'])
parser.add_argument('-zn', '--znz', dest='znz', type=str, required=True, help='Redshifts to use to construct the n(z)', choices=['zp', 'true_redshift_gal', 'observed_redshift_gal'])
parser.add_argument('-nz', '--nzbins', dest='nzbins', type=int, required=True, help='The number of galaxy equidistant tomographic bins')
parser.add_argument('-zmi', '--zmin', dest='zmin', type=float, required=True, help='Minimum redshift for the redshift binning')
parser.add_argument('-zma', '--zmax', dest='zmax', type=float, required=True, help='Maximum redshift for the redshift binning')

args = parser.parse_args()
dictargs = vars(args)
print("-----------------**  ARGUMENTS  **------------------------")
for keys in dictargs.keys():
    print(keys, " = ", dictargs[keys])

#-- Arguments
input_fits = args.input_fits
output_fits = args.output_fits
zselec = args.zselec
znz = args.znz
zmax = args.zmax
zmin = args.zmin
nzbins = args.nzbins

#-- Load the catalog
hdul = fits.open(input_fits)
print(hdul[1].header)

#-- Redshift binning
#- Edges of the bins
z_edges = np.linspace(zmin, zmax, nzbins+1)
print("z_edges : ", z_edges)

#- Selection
selecz=[]
for i in range(nzbins):
    #selection = (hdul[1].data['zp']>=z_edges[i]) & (hdul[1].data['zp']<=z_edges[i+1])
    selection = (hdul[1].data[zselec]>=z_edges[i]) & (hdul[1].data[zselec]<=z_edges[i+1])
    selecz.append(hdul[1].data[znz][selection])
hdul.close()

#-- Compute the histogram
nb = 400 # number of bins for the histogram
nz = np.zeros((nzbins,nb))
zb_centers = np.zeros((nzbins,nb))
for i in range(nzbins):
    nz[i], b = np.histogram(selecz[i], bins=nb, density=True)
    zb_centers[i] = (np.diff(b)/2.+b[:nb])
    print(b)
    
#-- Interpolate
numz = 10000 # length of z array
z_arr = np.linspace(0,3,numz)
nz_interp = np.zeros((nzbins, numz))
for i in range(nzbins):
    func = interpolate.InterpolatedUnivariateSpline(zb_centers[i], nz[i], k=1)
    nz_interp[i] = func(z_arr)
    nz_interp[i][nz_interp[i]<0] = 0
    
#-- Plot the interpolation
fig, ax = plt.subplots(1,1)
for i in range(nzbins):
    ax.plot(z_arr, nz_interp[i], ls='-', label=str(round(z_edges[i],2))+' > z > '+str(round(z_edges[i+1],2)))
ax.set_ylabel('$n(z)$')
ax.set_xlabel('$z$')
ax.legend(ncol=2)
plt.show()

#-- Save the n(z)
np.save('../data/'+output_fits+'_ZSELEC'+zselec+'_ZNZ'+znz+'_NZ'+str(nzbins)+'.npy', np.array([z_arr, nz_interp]))
