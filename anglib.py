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
- compute_fsky: Compute the fraction of sky covered by a given mask.

Dependencies:
- numpy
- healpy
- pymaster
- astropy
- time

Make sure to install the required libraries before using these functions.
"""

import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits
import time

def shear_map(ra, dec, g1, g2, ns, mask): 
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
    (array([...]), array([...]))
    """
    #- convert from deg to sr
    ra_rad = ra * (np.pi / 180)
    dec_rad = dec * (np.pi / 180)
    
    #- get the number of pixels for nside
    npix = hp.nside2npix(ns)
    
    #- get the pixel number for the given angles (need to change dec to theta pi/2.-dec)
    pix = hp.pixelfunc.ang2pix(ns, np.pi / 2. - dec_rad, ra_rad)

    #- map the shear values
    hpmapg1 = np.bincount(pix, minlength=npix, weights=g1)
    hpmapg2 = np.bincount(pix, minlength=npix, weights=g2)
    
    #- get the total number of galaxies
    ngal = dec_rad.size

    #- compute mean weight per visible pixel
    wbar = ngal / npix / np.mean(mask)

    #- normalize the shear values
    hpmapg1 /= wbar
    hpmapg2 /= wbar
    
    return hpmapg1, hpmapg2

def density_map(ra, dec, ns, mask):
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
    ra_rad = ra * (np.pi / 180)
    dec_rad = dec * (np.pi / 180)
    
    #- get the number of pixels for nside
    npix = hp.nside2npix(ns)
    
    #- get the pixel number for the given angles (need to change dec to theta pi/2.-dec)
    pix = hp.pixelfunc.ang2pix(ns, np.pi / 2. - dec_rad, ra_rad)
    
    #- get the hpmap (i.e. the number of particles per pixels) 
    hpmap = np.bincount(pix, weights=np.ones(dec.size), minlength=npix)
    
    #- get the total number of galaxies
    ngal = dec_rad.size
    
    #- Compute shotnoise
    sn = (4 * np.pi * compute_fsky(mask) ** 2) / ngal
    
    #- Compute nbar from all the pixels inside the mask and compute delta
    nbar = ngal / npix / np.mean(mask)
    hpmap = hpmap / nbar - mask
    
    return hpmap

def ang2map_radec(nns, ra, dec):
    """
    Convert a set of right ascension (RA) and declination (Dec) angles to a Healpix map.

    Parameters
    ----------
    nns : int
        NSIDE parameter for the Healpix map.
    ra : array-like
        Array containing right ascension angles in degrees.
    dec : array-like
        Array containing declination angles in degrees.

    Returns
    -------
    hpmap : array-like
        Healpix map of galaxy number counts.

    Examples
    --------
    >>> nns = 64
    >>> ra = np.array([10, 20, 30])
    >>> dec = np.array([45, 55, 65])
    >>> ang2map_radec(nns, ra, dec)
    array([...])
    """
    
    #- convert from deg to sr
    start = time.time()
    ra_rad = ra*(np.pi/180)
    dec_rad = dec*(np.pi/180)
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

def map2fld(hpmap1, hpmap2, mask):
    """
    Create NaMaster fields from Healpix maps.
    1) Get the edges of the mask
    2) Put to zero every pixel of the hp maps outside the mask
    3) Compute nbar from all the pixels inside the mask and Compute delta = n/nbar - 1
    4) Create a field from this map

    Parameters
    ----------
    hpmap1 : array-like
        First Healpix map.
    hpmap2 : array-like
        Second Healpix map.
    mask : array-like
        Mask defining the region of interest.

    Returns
    -------
    fld1 : NaMaster field
        NaMaster field corresponding to hpmap1.
    fld2 : NaMaster field
        NaMaster field corresponding to hpmap2.
    sn : float
        Shot noise computed based on the provided mask.

    Examples
    --------
    >>> hpmap1 = np.array([...])
    >>> hpmap2 = np.array([...])
    >>> mask = np.array([...])
    >>> map2fld(hpmap1, hpmap2, mask)
    (..., ..., ...)
    """
    
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

def compute_master(f_a, f_b, wsp, nns):
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
    nns : int
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
    
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)/hp.sphtfunc.pixwin(nns)**2
    cl_decoupled = wsp.decouple_cell(cl_coupled)

    return cl_decoupled

def ang2cl(nns, ra1, dec1, ra2, dec2, mask, wsp, nell_per_bin=1):
    """
    Perform the entire process to estimate Cl from a pair of (RA, Dec) sets.
    1) Project the tomographic bin of galaxy positions to a hp map. In other words, get the hpmap from (ra, dec)
    2) Get the associated galaxy overdensity fields f_1 and f_2 in a NaMaster format
    3) Estimate Cl of f_1 X f_2

    Parameters
    ----------
    nns : int
        NSIDE parameter for the Healpix maps.
    ra1 : array-like
        Right ascension angles for the first set.
    dec1 : array-like
        Declination angles for the first set.
    ra2 : array-like
        Right ascension angles for the second set.
    dec2 : array-like
        Declination angles for the second set.
    mask : array-like
        Mask defining the region of interest.
    wsp : nmt.NmtWorkspace
        NaMaster workspace containing pre-computed matrices.
    nell_per_bin : int, optional
        Number of ell values per bin (default is 1).

    Returns
    -------
    cl : array-like
        Estimated angular power spectrum.
    snl : array-like
        Decoupled shot noise power spectrum.

    Examples
    --------
    >>> ns = 64
    >>> ra1 = np.array([10, 20, 30])
    >>> dec1 = np.array([45, 55, 65])
    >>> ra2 = np.array([40, 50, 60])
    >>> dec2 = np.array([75, 85, 95])
    >>> mask = np.ones(hp.nside2npix(ns))
    >>> wsp = nmt.NmtWorkspace()
    >>> ang2cl(ns, ra1, dec1, ra2, dec2, mask, wsp)
    (array([...]), array([...]))
    """
    
    hpmap1 = ang2map_radec(nns, ra1, dec1)
    hpmap2 = ang2map_radec(nns, ra2, dec2)
    
    f1, f2, sn = map2fld(hpmap1, hpmap2, mask)
    
    start = time.time()
    cl = compute_master(f1, f2, wsp, nns)
    print(time.time()-start,'s for Cl estimation')
    
    #- decouple the shotnoise
    snl = np.array([ np.full(3 * nns, sn) ])/hp.sphtfunc.pixwin(nns)**2
    snl = wsp.decouple_cell(snl)
    
    return cl, snl

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

def compute_fsky(mask):
    """
    Compute the fraction of sky covered by a given mask.

    Parameters
    ----------
    mask : array-like
        Mask defining the region of interest.

    Returns
    -------
    fsky : float
        Fraction of the sky covered by the mask.

    Examples
    --------
    >>> mask = np.ones(12)
    >>> compute_fsky(mask)
    1.0
    """
    
    i_zeros = np.where(mask != 0)
    print('fsky=',float(i_zeros[0].size)/float(mask.size))
    return float(i_zeros[0].size)/float(mask.size)

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
    print("z_edges: ", z_edges)

    tomo_bins = []
    ngal_bin = []

    for i in selected_bins:
        if not (0 <= i < nbins):
            raise ValueError(f"Invalid bin index: {i}. It should be in the range [0, {nbins}).")

        start = time.time()
        selection = (hdul[1].data['zp'] >= z_edges[i]) & (hdul[1].data['zp'] <= z_edges[i + 1])
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