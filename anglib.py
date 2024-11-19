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

import os
import time
import itertools as it

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import healpy as hp
import pymaster as nmt
from astropy.io import fits
from astropy.table import Table
import twopoint
from statsmodels.stats.weightstats import DescrStatsW


import string_manager as stma



#---- Redshift binning ----#

def create_redshift_bins_complete(file_path, selected_bins, sample,
                                  division='EP_weights',
                                  nofz_redshift_type='true_redshift_gal',
                                  zmin=0.2, zmax=2.54, nbins=13):
    """
    Create redshift bins and corresponding galaxy catalogs from a FITS file
    for a specified set of bins.

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
        List of dictionaries containing galaxy catalog information for each
        selected redshift bin.
    ngal_bin : list
        Number of galaxies in each selected redshift bin.

    """
    print('\nTomographic binning for {} sample'.format(sample))
    dat = Table.read(file_path, format='fits')
    df = dat.to_pandas()

    # define z edges for the given division type (ED, EP, ...)
    if division == 'ED' :
        print('Dividing the sample in equi-distant tomographic bins.')
        z_edges = np.linspace(zmin, zmax, nbins + 1)

    elif division == 'EP_weight':
        print('Dividing the sample in equi-populated tomographic bins,')
        print('accounting for weights.')

        # cut data at appropriate redshift edges
        df = df[(df['zp'] >= zmin) & (df['zp'] < zmax)]

        # define redshift edges for binning
        weight = {'source':'she_weight',
                  'lens':'phz_weight'}
        wq = DescrStatsW(data=df['zp'], weights=df[weight[sample]])
        z_edges = wq.quantile(probs=np.linspace(0,1,nbins+1), return_pandas=False)

    # do the actual division in tomographic bins
    tomo_bins = []
    ngal_bin = []
    for i in selected_bins:
        if not (0 <= i < nbins):
            raise ValueError(f"Invalid bin index: {i}. It should be in the range [0, {nbins}).")
        print('- bin {}/{}'.format(i+1, nbins))
        selection = (df['zp'] >= z_edges[i]) & (df['zp'] <= z_edges[i+1])
        ngal_bin.append(df[nofz_redshift_type][selection].size)

        tbin = {
            'ra': df['ra'][selection],
            'dec': df['dec'][selection],
            'z': df[nofz_redshift_type][selection]
        }
        if sample == 'source':
            tbin['gamma1'] = df['gamma1'][selection]
            tbin['gamma2'] = df['gamma2'][selection]

        tomo_bins.append(tbin)

    return tomo_bins, ngal_bin

def build_nz(tbin, nb=400, nz=1000):

    nzbins = len(tbin)
    # find zmin and zmax
    zmin = tbin[0]['z'].min()
    zmax = tbin[0]['z'].max()
    for i in range(nzbins):
        if tbin[i]['z'].min() < zmin : zmin = tbin[i]['z'].min()
        if tbin[i]['z'].max() > zmax : zmax = tbin[i]['z'].max()

    # histogram quantities
    nofz = np.zeros((nzbins, nb))
    zb_centers = np.zeros((nzbins, nb))
    # interpolator quantities
    z_arr = np.linspace(zmin, zmax, nz)
    nz_interp = np.zeros((nzbins+1, nz))
    nz_interp[0] = z_arr
    for i in range(nzbins):
        # construct histogram
        nofz[i], b = np.histogram(tbin[i]['z'], bins=nb, density=True)
        zb_centers[i] = (np.diff(b)/2.+b[:nb])
        # interpolate
        func = InterpolatedUnivariateSpline(zb_centers[i], nofz[i], k=1)
        nz_interp[i+1] = func(z_arr)
        nz_interp[i+1][nz_interp[i+1]<0] = 0

    return nz_interp


#---- Maps ----#

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

    return [hpmapg1, hpmapg2]

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

    #- compute nbar from all the pixels inside the mask and compute delta
    nbar = ngal / npix / fsky
    hpmap = hpmap / nbar - mask

    return [hpmap]

def map2fld(hpmap, mask, lmax_bin):
    """
    Convert Healpix maps to NaMaster fields.

    Parameters
    ----------
    hpmap : list or array-like
        List or array containing the Healpix maps. If len(hpmap) is 2, the first map
        is treated as the negative component and the second as the positive component
        of the field. If len(hpmap) is 1, it's treated as a single map.
    mask : array-like
        Mask defining the footprint.

    Returns
    -------
    fld : NmtField
        NaMaster field object representing the input Healpix maps.

    Notes
    -----
    NaMaster (NmtField) handles fields for spherical harmonic analysis.
    This function prepares the fields by associating them with the provided mask.
    For len(hpmap) == 2, the negative and positive components are assigned accordingly.

    Examples
    --------
    >>> import numpy as np
    >>> import pymaster as nmt
    >>> mask = np.ones(12)  # Example mask
    >>> hpmap = [np.random.randn(12), np.random.randn(12)]  # Example Healpix maps
    >>> fld = map2fld(hpmap, mask)
    """

    if len(hpmap) == 2:
        fld = nmt.NmtField(mask, [-hpmap[0], hpmap[1]], lmax=lmax_bin)
    else:
        fld = nmt.NmtField(mask, [hpmap[0]], lmax=lmax_bin)

    return fld

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

#---- Cl's ----#

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

    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    i_lmax = cl_coupled.shape[1]
    cl_coupled /= pixwin(nside, depixelate)[:i_lmax]

    # decouple and bin
    cl_decoupled = wsp.decouple_cell(cl_coupled)

    return cl_decoupled[0]

def compute_coupled(f_a, f_b, bnmt, nside, depixelate):

    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    i_lmax = cl_coupled.shape[1]
    cl_coupled /= pixwin(nside, depixelate)[:i_lmax]

    # binning
    cl_binned = bnmt.bin_cell(np.array([cl_coupled[0]]))

    return cl_binned[0]

def linear_binning(lmax, lmin, bw):
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

    nbl = (lmax-lmin)//bw + 1
    elli = np.zeros(nbl, int)
    elle = np.zeros(nbl, int)

    for i in range(nbl):
        elli[i] = lmin + i*bw
        elle[i] = lmin + (i+1)*bw

    b = nmt.NmtBin.from_edges(elli, elle)
    return b

def log_binning_old(lmax, lmin, nbl):
    bin_edges = np.logspace(np.log10(lmin), np.log10(lmax), nbl)
    bin_edges = np.floor(bin_edges).astype(int)
    b = nmt.NmtBin.from_edges(bin_edges[:-1], bin_edges[1:])
    return b

def log_binning(lmax, lmin, nbl, w=None):
    op = np.log10
    inv = lambda x: 10**x

    bins = inv(np.linspace(op(lmin), op(lmax + 1), nbl + 1))
    ell = np.arange(lmin, lmax+1)
    i = np.digitize(ell, bins)-1
    if w is None:
        w = np.ones(ell.size)
    b = nmt.NmtBin(bpws=i, ells=ell, weights=w, lmax=lmax)
    return b

#---- Noise and pixels ----#

def shape_noise(tbin, fsky):
    """
    Compute the shape noise for a given tomographic bin.

    Parameters
    ----------
    tbin : dict
        Dictionary containing tomographic bin information.
    fsky : float
        Fraction of the sky covered by the survey.

    Returns
    -------
    shape_noise : float
        Shape noise estimate for the tomographic bin.

    Notes
    -----
    - The shape noise is calculated as the variance of the sum of squared ellipticities divided by the number of galaxies.
    - The formula is adjusted to account for the fraction of the sky surveyed.

    Examples
    --------
    >>> tbin = {'gamma1': np.array([0.1, 0.2, 0.3]),
    ...         'gamma2': np.array([0.2, 0.3, 0.4])}
    >>> fsky = 0.7
    >>> shape_noise(tbin, fsky)
    0.001

    """
    ngal = tbin['gamma1'].size
    var = ((tbin['gamma1']**2 + tbin['gamma2']**2)).sum() / ngal
    return var * (2 * np.pi * fsky**2) / ngal

def shot_noise(tbin, fsky):
    """
    Compute the shot noise for a given tomographic bin.

    Parameters
    ----------
    tbin : dict
        Dictionary containing tomographic bin information.
    fsky : float
        Fraction of the sky covered by the survey.

    Returns
    -------
    shot_noise : float
        Shot noise estimate for the tomographic bin.

    Notes
    -----
    - The shot noise is calculated as the inverse of the number density of galaxies.
    - The formula is adjusted to account for the fraction of the sky surveyed.

    Examples
    --------
    >>> tbin = {'dec': np.array([10, 20, 30])}
    >>> fsky = 0.5
    >>> shot_noise(tbin, fsky)
    0.007853981633974483
    """
    ngal = tbin['dec'].size
    return (4 * np.pi * fsky**2) / ngal

def compute_noise(tracer, tbin, fsky):
    if tracer == 'D':
        noise = shot_noise(tbin, fsky)
    if tracer == 'G':
        noise = shape_noise(tbin, fsky)

    return noise

def decouple_noise(noise_array, wsp, nside, depixelate):
    """
    Decouple the shot/shape noise power spectrum.

    Parameters
    ----------
    noise : float
        Shot/shape noise level.
    wsp : NaMaster workspace containing pre-computed matrices.
    nside : int
        NSIDE parameter for the Healpix maps.
    depixelate : bool
        Flag indicating whether to square the pixel window function.

    Returns
    -------
    snl_decoupled : array-like
        Decoupled shot/shape noise power spectrum.

    Notes
    -----
    Shot noise is the noise inherent in a measurement due to the discrete nature of the data.
    The shot noise power spectrum is decoupled using the provided NaMaster workspace (`wsp`)
    and the depixelation factor determined by the `depixelate` flag.

    Examples
    --------
    >>> noise = 0.1
    >>> wsp = nmt.NmtWorkspace()
    >>> nside = 64
    >>> depixelate = True
    >>> decouple_noise(noise, wsp, nside, depixelate)
    array([...])
    """

    i_lmax = noise_array.shape[1]

    snl = noise_array / pixwin(nside, depixelate)[:i_lmax]
    snl_decoupled = wsp.decouple_cell(snl)[0]

    return snl_decoupled

def couple_noise(noise_array, wsp, bnmt, fsky, nside, depixelate):

    # Not sure why but the debiasing agrees better with the one from
    # Heracles for WL like this...

    i_lmax = noise_array.shape[1]

    snl = noise_array / pixwin(nside, depixelate)[:i_lmax]
    # snl_coupled = wsp.couple_cell(snl)[0]
    snl_coupled = wsp.decouple_cell(snl)[0]*fsky

    # Bin the coupled noise
    # snl_coupled = bnmt.bin_cell(snl_coupled)[0]/fsky

    return snl_coupled

def debias(cl, noise, wsp, bnmt, fsky, nside, debias_bool, depixelate_bool, decouple_bool):
    """
    Debiases the angular power spectrum estimate by subtracting the decoupled noise.

    Parameters
    ----------
    cl : array-like
        Array containing the angular power spectrum estimate.
    noise : float
        Noise level estimate.
    w : object
        NaMaster workspace containing pre-computed matrices.
    nside : int
        NSIDE parameter for the Healpix maps.
    debias_bool : bool
        Boolean indicating whether to debias the spectrum.
    depixelate_bool : bool
        Boolean indicating whether to account for pixel window effects.

    Returns
    -------
    debiased_cl : array-like
        Debiased angular power spectrum estimate.

    Notes
    -----
    - If `debias_bool` is True, the decoupled noise is subtracted from the angular power spectrum.
    - The noise level estimate is used to compute the decoupled noise.
    - The pixel window effect is accounted for if `depixelate_bool` is True.

    Examples
    --------
    >>> cl = np.array([0.1, 0.2, 0.3])
    >>> noise = 0.05
    >>> w = nmt.NmtWorkspace()
    >>> nside = 64
    >>> debias_bool = True
    >>> depixelate_bool = True
    >>> debias(cl, noise, w, nside, debias_bool, depixelate_bool)
    array([0.09451773, 0.19451773, 0.29451773])
    """

    if debias_bool:

        lmax = wsp.wsp.bin.ell_max
        array_shape = wsp.get_bandpower_windows().shape[0]

        if array_shape == 1:
            noise_array = np.array([np.full(lmax+1, noise)])

        elif array_shape == 2:
            noise_array = np.array([np.full(lmax+1, noise),
                                    np.zeros(lmax+1)])
        elif array_shape == 4:
            noise_array = np.array([np.full(lmax+1, noise),
                                    np.zeros(lmax+1),
                                    np.zeros(lmax+1),
                                    np.zeros(lmax+1)])

        else :
            raise ValueError(f"Mixing matrix has weird shape ({array_shape})"
                             "It should be 1, 2 or 4")

        if decouple_bool:
            cl -= decouple_noise(noise_array, wsp, nside, depixelate_bool)

        else :
            cl -= couple_noise(noise_array, wsp, bnmt, fsky, nside, depixelate_bool)

    return cl

def pixwin(nside, depixelate):
    """
    Compute the pixel window function for a given NSIDE.

    Parameters
    ----------
    nside : int
        NSIDE parameter for the Healpix maps.
    depixelate : bool
        Flag indicating whether to square the pixel window function.

    Returns
    -------
    pixwin : array-like
        Pixel window function. If depixelate is True, the pixel window function is squared.

    Notes
    -----
    The pixel window function describes the suppression of power due to finite pixel size
    in the Healpix maps. If `depixelate` is True, the pixel window function is squared.
    This function provides flexibility in handling depixelation depending on the requirement.

    Examples
    --------
    >>> import healpy as hp
    >>> nside = 64
    >>> depixelate = True
    >>> pixwin(nside, depixelate)
    array([...])
    """
    pix_dic = {True: hp.sphtfunc.pixwin(nside) ** 2, False: np.ones(3 * nside)}
    return pix_dic[depixelate]

#---- Saving ----#
def save_twopoint(cls_dic, bnmt, nofz_dic, ngal_dic, zbins, probes, cross, outname):
    """
    Save the angular power spectra and number density data to a TwoPoint FITS file.

    Parameters
    ----------
    cls_dic : dict
        Dictionary containing the angular power spectra estimates.
    bnmt : object
        NaMaster binning object.
    nofz_dic : dic of array-like
        Array containing the redshifts and the n(z) for each bin.
    ngal_dic : dic of int or list of array-like
        Number density of galaxies.
    zbins : array-like
        Array containing the indices of redshift bins.
    probes : list
        List of probe types (e.g., ['GC', 'WL', 'GGL']).
    cross : bool
        Boolean indicating whether cross-correlations are computed.
    outname : str
        Output filename for the TwoPoint FITS file.

    Notes
    -----
    - The number density data is formatted from `nofz` and `ngal`.
    - The angular power spectra are formatted for each probe using `format_spectra_twopoint`.
    - All data is saved to a TwoPoint FITS file using the `to_fits` method.

    Examples
    --------
    >>> cls_dic = {'D0-D0': np.array([0.1, 0.2, 0.3]), 'D0-D1': np.array([0.2, 0.3, 0.4])}
    >>> bnmt = nmt.NmtBin.from_edges([0, 100, 200], [100, 200, 300])
    >>> nofz = (np.array([0.5, 1.0]), np.array([1.0, 2.0]))
    >>> ngal = 100
    >>> zbins = [0, 1]
    >>> probes = ['GC']
    >>> cross = False
    >>> outname = 'twopoint.fits'
    >>> save_twopoint(cls_dic, bnmt, nofz, ngal, zbins, probes, cross, outname)
    """
    # format n(z)
    kernels = []
    if 'GGL' in probes or 'WL' in probes:
        z_mid  = nofz_dic['source'][0]
        z_high = z_mid + np.diff(z_mid)[0]
        z_low  = z_mid - np.diff(z_mid)[0]
        nz_source = twopoint.NumberDensity('nz_source', zlow=z_low, z=z_mid, zhigh=z_high,
                                           nzs=nofz_dic['source'][1:], ngal=ngal_dic['source'])
        kernels.append(nz_source)

    if 'GGL' in probes or 'GC' in probes:
        z_mid  = nofz_dic['lens'][0]
        z_high = z_mid + np.diff(z_mid)[0]
        z_low  = z_mid - np.diff(z_mid)[0]
        nz_lens = twopoint.NumberDensity('nz_lens', zlow=z_low, z=z_mid, zhigh=z_high,
                                         nzs=nofz_dic['lens'][1:], ngal=ngal_dic['lens'])
        kernels.append(nz_lens)


    # format spectra
    spectra = []
    for p in probes :
        spectra.append(format_spectra_twopoint(cls_dic, bnmt, p, cross, zbins))

    # save twopoint fits file
    data = twopoint.TwoPointFile(spectra, kernels, None, None)
    data.to_fits(outname, overwrite=True)

def format_spectra_twopoint(cls_dic, bnmt, probe, cross, zbins):
    """
    Format the angular power spectra for a TwoPoint file.

    Parameters
    ----------
    cls_dic : dict
        Dictionary containing the angular power spectra estimates.
    bnmt : object
        NaMaster binning object.
    probe : str
        Probe type (e.g., 'GC', 'WL', 'GGL').
    cross : bool
        Boolean indicating whether cross-correlations are computed.
    zbins : array-like
        Array containing the indices of redshift bins.

    Returns
    -------
    spectrum : object
        SpectrumMeasurement object containing the formatted spectra.

    Notes
    -----
    - The formatted spectra are constructed as a SpectrumMeasurement object.
    - The angular power spectra estimates are retrieved from `cls_dic`.
    - Redshift bin indices are used to identify the bins in the spectra.

    Examples
    --------
    >>> cls_dic = {'D0-D0': np.array([0.1, 0.2, 0.3]), 'D0-D1': np.array([0.2, 0.3, 0.4])}
    >>> bnmt = nmt.NmtBin.from_edges([0, 100, 200], [100, 200, 300])
    >>> probe = 'GC'
    >>> cross = False
    >>> zbins = [0, 1]
    >>> format_spectra_twopoint(cls_dic, bnmt, probe, cross, zbins)
    <twopoint.core.SpectrumMeasurement object at ...>
    """

    # format ell binning
    ell = bnmt.get_effective_ells()
    nell = ell.size
    ell_max = np.zeros((nell))
    ell_min = np.zeros((nell))
    for i in range(bnmt.get_effective_ells().size):
        ell_max[i] = bnmt.get_ell_max(i)
        ell_min[i] = bnmt.get_ell_min(i)

    # initialise lists of info and quantities
    windows = "SAMPLE"
    multipole_bin = []
    multipole       = []
    bin1        = []
    bin2        = []
    multipole_min_arr = []
    multipole_max_arr = []
    cell = []

    # fill lists of info and quantities
    probe_iter = get_iter(probe, cross, zbins)
    for pi, pj in probe_iter:
        _, i = stma.mysplit(pi)
        _, j = stma.mysplit(pj)
        bin1.append(np.repeat(int(i), nell))
        bin2.append(np.repeat(int(j), nell))
        multipole.append(ell)
        multipole_min_arr.append(ell_min)
        multipole_max_arr.append(ell_max)
        multipole_bin.append(np.arange(nell))
        cell.append(cls_dic['{}-{}'.format(pi,pj)])

    # convert all the lists of vectors into long single vectors
    bin1 = np.concatenate(bin1)
    bin2 = np.concatenate(bin2)
    multipole = np.concatenate(multipole)
    multipole_min_arr = np.concatenate(multipole_min_arr)
    multipole_max_arr = np.concatenate(multipole_max_arr)
    multipole_bin = np.concatenate(multipole_bin)
    bins = (bin1, bin2)
    cell = np.concatenate(cell)

    # define types and nz_sample
    types = {'GC' : (twopoint.Types.galaxy_position_fourier,
                     twopoint.Types.galaxy_position_fourier),
             'WL' : (twopoint.Types.galaxy_shear_emode_fourier,
                     twopoint.Types.galaxy_shear_emode_fourier),
             'GGL' : (twopoint.Types.galaxy_position_fourier,
                      twopoint.Types.galaxy_shear_emode_fourier)}

    nz_sample = {'GC' : ('nz_lens', 'nz_lens'),
          'WL' : ('nz_source', 'nz_source'),
          'GGL' : ('nz_lens', 'nz_source')}

    # construct the SpectrumMeasurement object for the given probe
    spectrum = twopoint.SpectrumMeasurement('cell'+probe,
                                     bins,
                                     types[probe],
                                     nz_sample[probe],
                                     windows,
                                     multipole_bin,
                                     value=cell,
                                     angle=multipole,
                                     angle_min=multipole_min_arr,
                                     angle_max=multipole_max_arr,
                                     angle_unit='none')

    return spectrum

#---- Iterators ----#

def get_iter(probe, cross, zbins):
    """
    Get iterator for bin pairs based on the probe type and cross-correlation flag.

    Parameters
    ----------
    probe : str
        Probe type ('GC', 'WL', or 'GGL').
    cross : bool
        Flag indicating whether to compute cross-correlations.
    zbins : list
        List of redshift bin indices.

    Returns
    -------
    iterator : iterator
        Iterator yielding pairs of bin labels.

    Notes
    -----
    - The function constructs a dictionary `keymap` mapping probe types to bin labels.
    - Based on the cross-correlation flag, it selects the appropriate iterator method.
    - The resulting iterator provides bin pairs for the specified probe type and cross-correlation flag.

    Examples
    --------
    >>> probe = 'GC'
    >>> cross = ['GC']
    >>> zbins = [0, 1, 2]
    >>> get_iter(probe, cross, zbins)
    <itertools.combinations_with_replacement object at ...>
    """
    keymap = {'GC': ['D{}'.format(i+1) for i in zbins],
              'WL': ['G{}'.format(i+1) for i in zbins],
              'GGL': []}

    cross_selec = {
        True: it.combinations_with_replacement(keymap[probe], 2),
        False: zip(keymap[probe], keymap[probe])
    }

    iter_probe = {
        'GC': cross_selec['GC' in cross],
        'WL': cross_selec['WL' in cross],
        'GGL': it.product(keymap['GC'], keymap['WL'])
    }

    return iter_probe[probe]

def get_iter_nokeymap(probe, cross, zbins):
    """
    Get iterator for bin pairs based on the probe type and cross-correlation flag.

    Parameters
    ----------
    probe : str
        Probe type ('GC', 'WL', or 'GGL').
    cross : bool
        Flag indicating whether to compute cross-correlations.
    zbins : list
        List of redshift bin indices.

    Returns
    -------
    iterator : iterator
        Iterator yielding pairs of bin labels.

    Notes
    -----
    - The function constructs a dictionary `keymap` mapping probe types to bin labels.
    - Based on the cross-correlation flag, it selects the appropriate iterator method.
    - The resulting iterator provides bin pairs for the specified probe type and cross-correlation flag.

    Examples
    --------
    >>> probe = 'GC'
    >>> cross = ['GC']
    >>> zbins = [0, 1, 2]
    >>> get_iter(probe, cross, zbins)
    <itertools.combinations_with_replacement object at ...>
    """
    keymap = {'GC': ['{}'.format(i+1) for i in zbins],
              'WL': ['{}'.format(i+1) for i in zbins],
              'GGL': []}

    cross_selec = {
        True: it.combinations_with_replacement(keymap[probe], 2),
        False: zip(keymap[probe], keymap[probe])
    }

    iter_probe = {
        'GC': cross_selec['GC' in cross],
        'WL': cross_selec['WL' in cross],
        'GGL': it.product(keymap['GC'], keymap['WL'])
    }

    return iter_probe[probe]

#---- Workspaces ----#

def create_workspaces(bin_scheme, mask, wkspce_name, probe):
    spin_probe_dic = {
        'GC'  : (0,0),
        'WL'  : (2,2),
        'GGL' : (0,2)}

    return coupling_matrix(bin_scheme, mask, wkspce_name, *spin_probe_dic[probe])

def coupling_matrix(bin_scheme, mask, wkspce_name, spin1, spin2):
    """
    Compute the mixing matrix for coupling spherical harmonic modes using
    the provided binning scheme and mask.

    Parameters:
    -----------
    bin_scheme : nmt_bins
        A binning scheme object defining the bins for the coupling matrix.

    mask : nmt_field
        A mask object defining the regions of the sky to include in the computation.

    wkspce_name : str
        The file name for storing or retrieving the computed workspace containing
        the coupling matrix.

    Returns:
    --------
    nmt_workspace
        A workspace object containing the computed coupling matrix.

    Notes:
    ------
    This function computes the coupling matrix necessary for the pseudo-Cl power
    spectrum estimation using the NmtField and NmtWorkspace objects from the
    Namaster library.

    If the workspace file specified by 'wkspce_name' exists, the function reads
    the coupling matrix from the file. Otherwise, it computes the matrix and
    writes it to the file.

    Example:
    --------
    # Generate a linear binning scheme for an NSIDE of 64, starting from l=10, with bin width of 20
    bin_scheme = linear_lmin_binning(NSIDE=64, lmin=10, bw=20)

    # Define the mask
    mask = nmt.NmtField(mask, [mask])

    # Compute the coupling matrix and store it in 'coupling_matrix.bin'
    coupling_matrix = coupling_matrix(bin_scheme, mask, 'coupling_matrix.bin')
    """
    print('Compute the mixing matrix')
    start = time.time()
    fmask1 = nmt.NmtField(mask, None, lmax=bin_scheme.lmax, spin=spin1) # nmt field with only the mask
    fmask2 = nmt.NmtField(mask, None, lmax=bin_scheme.lmax, spin=spin2) # nmt field with only the mask
    w = nmt.NmtWorkspace()
    if os.path.isfile(wkspce_name):
        print('Mixing matrix has already been calculated and is in the workspace file : ', wkspce_name, '. Read it.')
        w.read_from(wkspce_name)
    else :
        print('The file : ', wkspce_name, ' does not exists. Calculating the mixing matrix and writing it.')
        w.compute_coupling_matrix(fmask1, fmask2, bin_scheme)
        w.write_to(wkspce_name)
    print('Done computing the mixing matrix. It took ', time.time()-start, 's.')
    return w

#---- Utils ----#
def probe_ref_mapping(probe):
    mapping = {
        'GC' : 'galaxy_cl',
        'WL' : 'shear_cl',
        'GGL' : 'galaxy_shear_cl'}

    return mapping[probe]
