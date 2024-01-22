"""
anglib.py : Library for photometric analysis in harmonic space

This Python file contains a set of functions designed for estimating 2-point statistics
for photometric analysis in cosmology. The functions utilize the Healpix framework,
NaMaster (a Python wrapper for MASTER), and other scientific computing libraries.

Functions:
- ang2map_radec: Convert RA and Dec angles to a Healpix map of galaxy number counts.
- map2fld: Create NaMaster fields from Healpix maps for a given mask.
- compute_master: Compute angular power spectrum (Cls) from a pair of NaMaster fields.
- ang2cl: Perform the entire process to estimate Cl from (RA, Dec) sets using NaMaster.
- edges_binning: Define an ell binning for a given NSIDE, lmin, and bin width.
- create_redshift_bins: Create redshift bins and corresponding galaxy catalogs from a FITS file for a specified set of bins.

Dependencies:
- numpy
- healpy
- pymaster
- astropy
- time

"""

import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits

import time
import itertools as it

def create_redshift_bins(file_path, selected_bins, zmin=0.2, zmax=2.54, nbins=13):
    """
    Create redshift bins and corresponding galaxy catalogs from a FITS file for a specified set of bins.

    Parameters
    ----------
    file_path : str
        Path to the FITS file containing galaxy data.
    selected_bins : list
        List of indices specifying the redshift bins to consider.
    zmin : float, optional
        Minimum redshift for the bins (default is 0.2).
    zmax : float, optional
        Maximum redshift for the bins (default is 2.54).
    nbins : int, optional
        Number of redshift bins (default is 13).

    Returns
    -------
    tomo_bins : list
        List of dictionaries containing galaxy catalog information for each selected redshift bin.
    ngal_bin : list
        Number of galaxies in each selected redshift bin.

    Examples
    --------
    >>> file_path = 'galaxy_data.fits'
    >>> selected_bins = [0, 1, 2]
    >>> create_redshift_bins(file_path, selected_bins)
    ([{'gamma1': array([...]),
       'gamma2': array([...]),
       'ra': array([...]),
       'dec': array([...]),
       'z': array([...])},
      {'gamma1': array([...]),
       'gamma2': array([...]),
       'ra': array([...]),
       'dec': array([...]),
       'z': array([...])},
      {'gamma1': array([...]),
       'gamma2': array([...]),
       'ra': array([...]),
       'dec': array([...]),
       'z': array([...])}], [100, 150, 200])
    """
    
    hdul = fits.open(file_path)
    
    z_edges = np.linspace(zmin, zmax, nbins + 1)

    tomo_bins = []
    ngal_bin = []
    print('Dividing the sample in equi-distant tomographic bins')
    for i in selected_bins:
        if not (0 <= i < nbins):
            raise ValueError(f"Invalid bin index: {i}. It should be in the range [0, {nbins}).")
        print('- bin {}/{}'.format(i, nbins))
        selection = (hdul[1].data['zp'] >= z_edges[i]) & (hdul[1].data['zp'] <= z_edges[i+1])
        ngal_bin.append(hdul[1].data['observed_redshift_gal'][selection].size)

        tbin = {
            'gamma1': hdul[1].data['gamma1'][selection],
            'gamma2': hdul[1].data['gamma2'][selection],
            'ra': hdul[1].data['ra_gal'][selection],
            'dec': hdul[1].data['dec_gal'][selection],
            'z': hdul[1].data['observed_redshift_gal'][selection]
        }
        tomo_bins.append(tbin)

    hdul.close()
    
    return tomo_bins, ngal_bin

def shear_map(tbin, nside, mask): 
    """
    Map shear values to Healpix maps based on galaxy positions and observed ellipticities.

    Parameters
    ----------
    ra : array-like
        Array containing right ascension angles in degrees.
    dec : array-like
        Array containing declination angles in degrees.
    g1 : array-like
        Array containing the first component of shear values.
    g2 : array-like
        Array containing the second component of shear values.
    ns : int
        NSIDE parameter for the Healpix map.
    mask : array-like
        Mask defining the footprint.

    Returns
    -------
    hpmapg1 : array-like
        Healpix map of the first component of shear values.
    hpmapg2 : array-like
        Healpix map of the second component of shear values.

    Examples
    --------
    >>> ra = np.array([10, 20, 30])
    >>> dec = np.array([45, 55, 65])
    >>> g1 = np.array([0.1, 0.2, 0.3])
    >>> g2 = np.array([0.2, 0.3, 0.4])
    >>> ns = 64
    >>> mask = np.ones(hp.nside2npix(ns))
    >>> shear_map(ra, dec, g1, g2, ns, mask)
    [array([...]), array([...])]
    """
    #- convert from deg to sr
    ra_rad = tbin['ra'] * (np.pi / 180)
    dec_rad = tbin['dec'] * (np.pi / 180)
    
    #- get the number of pixels for nside
    npix = hp.nside2npix(nside)
    
    #- get the pixel number for the given angles (need to change dec to theta pi/2.-dec)
    pix = hp.pixelfunc.ang2pix(nside, np.pi / 2. - dec_rad, ra_rad)

    #- map the shear values
    hpmapg1 = np.bincount(pix, minlength=npix, weights=tbin['gamma1'])
    hpmapg2 = np.bincount(pix, minlength=npix, weights=tbin['gamma2'])
    
    #- get the total number of galaxies
    ngal = dec_rad.size

    #- compute mean weight per visible pixel
    fsky = np.mean(mask)
    nbar = ngal / npix / fsky
    
    #- compute shape-noise
    shapenoise = shape_noise(tbin, fsky)

    #- normalize the shear values
    hpmapg1 = hpmapg1/nbar
    hpmapg2 = hpmapg2/nbar
    
    return [hpmapg1, hpmapg2], shapenoise

def density_map(tbin, nside, mask):
    """
    Map galaxy number density to Healpix maps based on galaxy positions.

    Parameters
    ----------
    ra : array-like
        Array containing right ascension angles in degrees.
    dec : array-like
        Array containing declination angles in degrees.
    ns : int
        NSIDE parameter for the Healpix map.
    mask : array-like
        Mask defining the footprint.

    Returns
    -------
    hpmap : array-like
        Healpix map of galaxy number density.

    Examples
    --------
    >>> ra = np.array([10, 20, 30])
    >>> dec = np.array([45, 55, 65])
    >>> ns = 64
    >>> mask = np.ones(hp.nside2npix(ns))
    >>> density_map(ra, dec, ns, mask)
    array([...])
    """
    
    #- convert from deg to sr
    ra_rad = tbin['ra'] * (np.pi / 180)
    dec_rad = tbin['dec'] * (np.pi / 180)
    
    #- get the number of pixels for nside
    npix = hp.nside2npix(nside)
    
    #- get the pixel number for the given angles (need to change dec to theta pi/2.-dec)
    pix = hp.pixelfunc.ang2pix(nside, np.pi / 2. - dec_rad, ra_rad)
    
    #- get the hpmap (i.e. the number of particles per pixels) 
    hpmap = np.bincount(pix, weights=np.ones(dec_rad.size), minlength=npix)
    
    #- get the total number of galaxies
    ngal = dec_rad.size
    
    #- get sky fraction
    fsky = np.mean(mask)
    
    #- compute shot-noise
    shotnoise = shot_noise(tbin, fsky)
    
    #- compute nbar from all the pixels inside the mask and compute delta
    nbar = ngal / npix / fsky
    hpmap = hpmap / nbar - mask
    
    return [hpmap], shotnoise

def shape_noise(tbin, fsky):
    ngal = tbin['gamma1'].size
    var = ((tbin['gamma1']**2 + tbin['gamma2']**2)).sum()/ngal 
    return var * ( 2*np.pi * fsky**2) / ngal

def shot_noise(tbin, fsky):
    ngal = tbin['dec'].size
    return (4*np.pi * fsky**2) / ngal

def get_map_for_probes(probes):
    """
    Get the required functions to compute maps based on the selected probes.

    Parameters
    ----------
    probes : list
        List of probes to consider. Possible values: 'GC' (galaxy clustering),
        'WL' (weak lensing), 'GGL' (galaxy-galaxy lensing).

    Returns
    -------
    maps : list
        List of function computing the maps based on the selected probes.

    Examples
    --------
    >>> probes = ['GC', 'WL']
    >>> get_map_for_probes(probes)
    [density_map, shear_map]

    """
    maps = []
    keys  = []

    if 'GC' in probes or 'GGL' in probes:
        gc_map = density_map
        maps.append(gc_map)
        keys.append('D')

    if 'WL' in probes or 'GGL' in probes:
        wl_map = shear_map
        maps.append(wl_map)
        keys.append('G')

    return maps, keys

def get_iter(probe, cross, zbins):
    
    keymap = {'GC'  :  ['D{}'.format(i) for i in zbins],
              'WL'  :  ['G{}'.format(i) for i in zbins],
              'GGL' :  []}
    
    cross_selec = {True  : it.combinations_with_replacement(keymap[probe], 2),
                   False : zip(keymap[probe], keymap[probe])} 

    iter_probe = {'GC'  : cross_selec['GC' in cross],
                  'WL'  : cross_selec['WL' in cross],
                  'GGL' : it.product(keymap['GC'], keymap['WL'])}
    
    return iter_probe[probe]

def map2fld(hpmap, mask):
    if len(hpmap) == 2:
        fld = nmt.NmtField(mask, [-hpmap[0], hpmap[1]])
    else:
        fld = nmt.NmtField(mask, [hpmap[0]])
        
    return fld

def pixwin(nside, depixelate):
    pix_dic = {True  : hp.sphtfunc.pixwin(nside)**2,
               False : 1}
    return pix_dic[depixelate]

def compute_master(f_a, f_b, wsp, nside, depixelate):
    """
    Compute the angular power spectrum (Cls) from a pair of NaMaster fields.

    Parameters
    ----------
    f_a : nmt.NmtField
        NaMaster field for the first map.
    f_b : nmt.NmtField
        NaMaster field for the second map.
    wsp : nmt.NmtWorkspace
        NaMaster workspace containing pre-computed matrices.
    nside : int
        NSIDE parameter for the Healpix maps.

    Returns
    -------
    cl_decoupled : array-like
        Decoupled angular power spectrum.

    Examples
    --------
    >>> import pymaster as nmt
    >>> ns = 64
    >>> f_a = nmt.NmtField(mask, [hpmap1])
    >>> f_b = nmt.NmtField(mask, [hpmap2])
    >>> wsp = nmt.NmtWorkspace()
    >>> compute_master(f_a, f_b, wsp, ns)
    array([...])
    """
    
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)/pixwin(nside, depixelate)
    cl_decoupled = wsp.decouple_cell([cl_coupled[0]])

    return cl_decoupled[0]

def decouple_noise(noise, wsp, nside, depixelate):
    snl = np.array([ np.full(3 * nside, noise) ])/pixwin(nside, depixelate)
    return wsp.decouple_cell(snl)[0]

def edges_binning(NSIDE, lmin, bw):
    """
    Define an ell binning for a given NSIDE, lmin, and bin width.

    Parameters
    ----------
    NSIDE : int
        NSIDE parameter for the Healpix maps.
    lmin : int
        Minimum ell value for the binning.
    bw : int
        Bin width for the ell values.

    Returns
    -------
    b : nmt.NmtBin
        NaMaster binning object.

    Examples
    --------
    >>> nside = 64
    >>> lmin = 10
    >>> bw = 5
    >>> edges_binning(nside, lmin, bw)
    nmt.NmtBin object with 13 bins: [(10, 15), (15, 20), ..., (60, 65)]
    """

    lmax = 2*NSIDE
    nbl = (lmax-lmin)//bw + 1
    elli = np.zeros(nbl, int)
    elle = np.zeros(nbl, int)

    for i in range(nbl):
        elli[i] = lmin + i*bw
        elle[i] = lmin + (i+1)*bw

    b = nmt.NmtBin.from_edges(elli, elle)
    return b

