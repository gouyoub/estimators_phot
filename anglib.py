"""
anglib.py : Library for photometric analysis in harmonic space

This Python file contains a set of functions designed for estimating 2-point statistics
for photometric analysis in cosmology. The functions utilize the Healpix framework,
NaMaster (a Python wrapper for MASTER), and other scientific computing libraries.

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


from general import mysplit



#---- Redshift binning ----#

def create_redshift_bins(file_path, columns, selected_bins, sample,
                                  division='EP_weights',
                                  nofz_redshift_type='true_redshift_gal',
                                  zmin=0.2, zmax=2.54, nbins=13):
    """
    Create redshift bins and corresponding galaxy catalogs from a FITS file
    for a specified set of bins.

    This function processes galaxy data from a FITS file and divides it into
    tomographic redshift bins. The bins can be either equidistant or equi-populated,
    depending on the specified division method. For each selected bin, a catalog of
    galaxies is created, which includes relevant attributes like redshift, position,
    and optionally shear components.

    Parameters
    ----------
    file_path : str
        Path to the FITS file containing galaxy data.
    columns : dict
        Dictionary mapping column names in the FITS file to their roles. Expected keys:
        - 'zp': Column for photometric or estimated redshift.
        - 'ra': Column for right ascension.
        - 'dec': Column for declination.
        - 'gamma1': Column for the first shear component (required for 'source' sample).
        - 'gamma2': Column for the second shear component (required for 'source' sample).
        - 'she_weight': Shear weight column (used for equi-populated bins if `sample='source'`).
        - 'phz_weight': Photometric weight column (used for equi-populated bins if `sample='lens'`).
    selected_bins : list of int
        List of indices specifying the redshift bins to include in the output.
    sample : str
        Type of galaxy sample, either 'source' or 'lens'. Determines which weights and
        additional properties (e.g., shear) are included.
    division : str, optional
        Method for dividing redshift bins:
        - 'ED': Equidistant bins.
        - 'EP_weights': Equi-populated bins, accounting for weights.
        Default is 'EP_weights'.
    nofz_redshift_type : str, optional
        Column name for the redshift to be used in the output catalogs.
        Default is 'true_redshift_gal'.
    zmin : float, optional
        Minimum redshift for the bins (default is 0.2).
    zmax : float, optional
        Maximum redshift for the bins (default is 2.54).
    nbins : int, optional
        Number of total redshift bins (default is 13).

    Returns
    -------
    tomo_bins : list of dict
        List of dictionaries, each corresponding to a selected redshift bin.
        Each dictionary contains:
        - 'ra': Right ascension of galaxies in the bin.
        - 'dec': Declination of galaxies in the bin.
        - 'z': Redshift of galaxies in the bin.
        - 'gamma1' and 'gamma2' (if `sample='source'`): Shear components for galaxies.
    ngal_bin : list of int
        Number of galaxies in each selected redshift bin.

    Raises
    ------
    ValueError
        If any of the selected bin indices are out of the valid range [0, nbins).
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
        df = df[(df[columns['zp']] >= zmin) & (df[columns['zp']] < zmax)]

        # define redshift edges for binning
        weight = {'source':'she_weight',
                  'lens':'phz_weight'}
        wq = DescrStatsW(data=df[columns['zp']], weights=df[columns[weight[sample]]])
        z_edges = wq.quantile(probs=np.linspace(0,1,nbins+1), return_pandas=False)

    elif division == 'EP_sharpcut_lenses':
        print('Dividing the sample in equi-populated tomographic bins,')
        print('sharp cutting from the weights for the lenses.')

        # cut data at appropriate redshift edges
        if sample == 'lens':
            df = df[(df[columns['zp']] >= zmin) & (df[columns['zp']] < zmax)
                    & (df['phz_weight'] >= 0.5)]
        elif sample == 'source':
            df = df[(df[columns['zp']] >= zmin) & (df[columns['zp']] < zmax)]

        # define redshift edges for binning
        weight = {'source':'she_weight',
                  'lens':'phz_weight'}
        wq = DescrStatsW(data=df[columns['zp']])
        z_edges = wq.quantile(probs=np.linspace(0,1,nbins+1), return_pandas=False)

    # do the actual division in tomographic bins
    tomo_bins = []
    ngal_bin = []
    for i in selected_bins:
        if not (0 <= i < nbins):
            raise ValueError(f"Invalid bin index: {i}. It should be in the range [0, {nbins}).")
        print('- bin {}/{}'.format(i+1, nbins))
        selection = (df[columns['zp']] >= z_edges[i]) & (df[columns['zp']] <= z_edges[i+1])
        ngal_bin.append(df[nofz_redshift_type][selection].size)

        tbin = {
            'ra': df[columns['ra']][selection],
            'dec': df[columns['dec']][selection],
            'z': df[nofz_redshift_type][selection]
        }
        if sample == 'source':
            tbin['gamma1'] = df[columns['gamma1']][selection]
            tbin['gamma2'] = df[columns['gamma2']][selection]

        tomo_bins.append(tbin)

    return tomo_bins, ngal_bin, z_edges

def build_nz(tbin, nb=400, nz=1000):
    """
    Build normalized redshift distributions and interpolated values for a set of tomographic bins.

    This function computes the redshift distributions (n(z)) for a given set of tomographic bins
    using histogramming and interpolation. It also provides interpolated n(z) values over a
    specified range of redshifts for each bin.

    Parameters
    ----------
    tbin : list of dict
        List of dictionaries, each representing a tomographic bin. Each dictionary should have:
        - 'z': Array-like, containing the redshift values of galaxies in the bin.
    nb : int, optional
        Number of histogram bins for constructing the redshift distribution. Default is 400.
    nz : int, optional
        Number of interpolation points for the redshift range. Default is 1000.

    Returns
    -------
    nz_interp : numpy.ndarray
        A 2D array of shape `(nzbins + 1, nz)`, where:
        - The first row contains the interpolated redshift values (z) over the range [zmin, zmax].
        - Subsequent rows contain the interpolated n(z) values for each tomographic bin.
    """
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
    tbin : dict
        Dictionary containing galaxy catalog data with the following keys:
        - 'ra': Array of right ascension angles in degrees.
        - 'dec': Array of declination angles in degrees.
        - 'gamma1': Array of the first component of shear values.
        - 'gamma2': Array of the second component of shear values.
    nside : int
        NSIDE parameter for the Healpix map.
    mask : array-like
        Mask defining the footprint.

    Returns
    -------
    hpmapg1 : array-like
        Healpix map of the first component of shear values.
    hpmapg2 : array-like
        Healpix map of the second component of shear values.
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

    #- normalize the shear values
    hpmapg1 = hpmapg1/nbar
    hpmapg2 = hpmapg2/nbar

    return [hpmapg1, hpmapg2]

def density_map(tbin, nside, mask):
    """
    Map galaxy number density to Healpix maps based on galaxy positions.

    Parameters
    ----------
    tbin : dict
        Dictionary containing galaxy catalog data with the following keys:
        - 'ra': Array of right ascension angles in degrees.
        - 'dec': Array of declination angles in degrees.
    nside : int
        NSIDE parameter for the Healpix map.
    mask : array-like
        Mask defining the footprint.

    Returns
    -------
    hpmap : array-like
        Healpix map of galaxy number density.
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
        List or array containing the Healpix maps. If `len(hpmap)` is 2, the first map
        is treated as the negative component and the second as the positive component
        of the field. If `len(hpmap)` is 1, it's treated as a single map.
    mask : array-like
        Mask defining the footprint.
    lmax_bin : int
        Maximum multipole used for the field.

    Returns
    -------
    fld : NmtField
        NaMaster field object representing the input Healpix maps.
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
        List of functions for computing the maps based on the selected probes.
    keys : list
        List of keys ('D' for density, 'G' for shear) corresponding to the selected probes.
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
    depixelate : bool
        Flag indicating whether to account for pixel window effects.

    Returns
    -------
    cl_decoupled : array-like
        Decoupled angular power spectrum.
    """
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    i_lmax = cl_coupled.shape[1]
    cl_coupled /= pixwin(nside, depixelate)[:i_lmax]
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled[0]

def compute_coupled(f_a, f_b, bnmt, nside, depixelate):
    """
    Compute and bin the coupled angular power spectrum (Cls) from a pair of NaMaster fields.

    Parameters
    ----------
    f_a : nmt.NmtField
        NaMaster field for the first map.
    f_b : nmt.NmtField
        NaMaster field for the second map.
    bnmt : nmt.NmtBin
        Binning scheme for the power spectrum.
    nside : int
        NSIDE parameter for the Healpix maps.
    depixelate : bool
        Flag indicating whether to account for pixel window effects.

    Returns
    -------
    cl_binned : array-like
        Binned coupled angular power spectrum.
    """
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    i_lmax = cl_coupled.shape[1]
    cl_coupled /= pixwin(nside, depixelate)[:i_lmax]
    cl_binned = bnmt.bin_cell(np.array([cl_coupled[0]]))
    return cl_binned[0]

def linear_binning(lmax, lmin, bw):
    """
    Define a linear ell binning scheme.

    Parameters
    ----------
    lmax : int
        Maximum ell value for the binning.
    lmin : int
        Minimum ell value for the binning.
    bw : int
        Bin width for the ell values.

    Returns
    -------
    b : nmt.NmtBin
        NaMaster binning object with linear bins.
    """
    nbl = (lmax - lmin) // bw + 1
    elli = np.arange(lmin, lmin + nbl * bw, bw)
    elle = elli + bw
    b = nmt.NmtBin.from_edges(elli, elle)
    return b

def log_binning(lmax, lmin, nbl, w=None):
    """
    Define a logarithmic ell binning scheme with optional weights.

    Parameters
    ----------
    lmax : int
        Maximum ell value for the binning.
    lmin : int
        Minimum ell value for the binning.
    nbl : int
        Number of bins.
    w : array-like, optional
        Weights for the ell values.

    Returns
    -------
    b : nmt.NmtBin
        NaMaster binning object with logarithmic bins.
    """
    op = np.log10
    inv = lambda x: 10**x
    bins = inv(np.linspace(op(lmin), op(lmax + 1), nbl + 1))
    ell = np.arange(lmin, lmax + 1)
    i = np.digitize(ell, bins) - 1
    if w is None:
        w = np.ones(ell.size)
    b = nmt.NmtBin(bpws=i, ells=ell, weights=w, lmax=lmax)
    return b

#---- Noise and pixels ----#

def shape_var(tbin):
    ngal = tbin['gamma1'].size
    var = ((tbin['gamma1']**2 + tbin['gamma2']**2)).sum() / ngal
    return var

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
    """
    ngal = tbin['gamma1'].size
    var = shape_var(tbin)
    return var * (2 * np.pi * fsky) / ngal

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
    """
    ngal = tbin['dec'].size
    return (4 * np.pi * fsky) / ngal

def compute_noise(tracer, tbin, fsky):
    """
    Compute the noise level for a specific tracer.

    Parameters
    ----------
    tracer : str
        Type of tracer ('D' for density or 'G' for shear).
    tbin : dict
        Dictionary containing tomographic bin information.
    fsky : float
        Fraction of the sky covered by the survey.

    Returns
    -------
    noise : float
        Noise level estimate for the tracer.
    """
    if tracer == 'D':
        noise = shot_noise(tbin, fsky)
    if tracer == 'G':
        noise = shape_noise(tbin, fsky)
    return noise

def decouple_noise(noise_array, wsp, fsky, nside, depixelate):
    """
    Decouples the noise power spectrum using a NaMaster workspace.

    Parameters
    ----------
    noise_array : array-like
        Array representing the noise power spectrum for each mode.
        The shape must be consistent with the expected input for the NaMaster workspace.
    wsp : nmt.NmtWorkspace
        NaMaster workspace containing precomputed coupling matrices for decoupling.
    nside : int
        HEALPix NSIDE parameter for the maps.
    depixelate : bool
        If True, squares the pixel window function to account for pixelation effects.

    Returns
    -------
    snl_decoupled : array-like
        The decoupled noise power spectrum, corrected for pixelation effects if specified.

    """

    i_lmax = noise_array.shape[1]

    snl = noise_array / pixwin(nside, depixelate)[:i_lmax]
    snl_decoupled = wsp.decouple_cell(snl)[0]*fsky # multiplying by fsky to compensate the decoupling

    return snl_decoupled

def couple_noise(noise_array, wsp, bnmt, fsky, nside, depixelate):
    """
    Couples the noise power spectrum using a NaMaster workspace and applies sky fraction scaling.

    Parameters
    ----------
    noise_array : array-like
        Array representing the noise power spectrum for each mode.
    wsp : nmt.NmtWorkspace
        NaMaster workspace containing precomputed coupling matrices.
    bnmt : object
        Binning scheme used to bin the coupled noise (if applicable).
    fsky : float
        Sky fraction covered by the mask, used for scaling the coupled noise.
    nside : int
        HEALPix NSIDE parameter for the maps.
    depixelate : bool
        If True, squares the pixel window function to account for pixelation effects.

    Returns
    -------
    snl_coupled : array-like
        The coupled noise power spectrum, scaled by the sky fraction and optionally binned.

    """

    # Not sure why but the debiasing agrees better with the one from
    # Heracles for WL like this...

    i_lmax = noise_array.shape[1]

    snl = noise_array / pixwin(nside, depixelate)[:i_lmax]
    # snl_coupled = wsp.couple_cell(snl)[0]
    snl_coupled = wsp.decouple_cell(snl)[0]*fsky**2 # multiplying by fsky to compensate the decoupling
                                                    # and another time to account for the coupling of the Cl's

    # Bin the coupled noise
    # snl_coupled = bnmt.bin_cell(snl_coupled)[0]/fsky

    return snl_coupled

def debias(cl, noise, wsp, bnmt, fsky, nside, debias_bool, depixelate_bool, decouple_bool):
    """
    Removes noise bias from the angular power spectrum estimate.

    Parameters
    ----------
    cl : array-like
        Angular power spectrum estimate to be debiased.
    noise : float
        Noise level estimate, assumed constant across multipoles.
    wsp : nmt.NmtWorkspace
        NaMaster workspace object containing precomputed mixing matrices.
    bnmt : object
        Binning scheme used for the angular power spectrum (e.g., linear or logarithmic).
    fsky : float
        Sky fraction covered by the mask.
    nside : int
        HEALPix NSIDE parameter of the maps.
    debias_bool : bool
        If True, the noise bias is subtracted from the spectrum.
    depixelate_bool : bool
        If True, accounts for the pixel window effect in the noise subtraction.
    decouple_bool : bool
        If True, performs noise subtraction in decoupled space; otherwise, in coupled space.

    Returns
    -------
    cl : array-like
        Debiased angular power spectrum estimate.

    Raises
    ------
    ValueError
        If the shape of the mixing matrix in `wsp` is not 1, 2, or 4.

    Notes
    -----
    - The noise bias is estimated based on the input `noise` level and the NaMaster workspace (`wsp`).
    - Depending on `decouple_bool`, noise subtraction is performed in either decoupled or coupled space.
    - If `depixelate_bool` is True, pixel window effects are accounted for during noise subtraction.

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
            cl -= decouple_noise(noise_array, wsp, fsky, nside, depixelate_bool)

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
    """
    pix_dic = {True: hp.sphtfunc.pixwin(nside) ** 2, False: np.ones(3 * nside)}
    return pix_dic[depixelate]

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
    nofz_dic : dict of array-like
        Array containing the redshifts and the n(z) for each bin.
    ngal_dic : dict of int or list of array-like
        Number density of galaxies.
    zbins : array-like
        Array containing the indices of redshift bins.
    probes : list
        List of probe types (e.g., ['GC', 'WL', 'GGL']).
    cross : bool
        Boolean indicating whether cross-correlations are computed.
    outname : str
        Output filename for the TwoPoint FITS file.
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
        _, i = mysplit(pi)
        _, j = mysplit(pj)
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
    """
    Create workspaces for computing coupling matrices based on the probe type.

    Parameters
    ----------
    bin_scheme : nmt_bins
        A binning scheme object defining the bins for the coupling matrix.
    mask : nmt_field
        A mask object defining the regions of the sky to include in the computation.
    wkspce_name : str
        The filename for storing or retrieving the computed workspace.
    probe : str
        Probe type ('GC', 'WL', or 'GGL').

    Returns
    -------
    nmt_workspace
        Workspace object containing the computed coupling matrix.
    """
    spin_probe_dic = {
        'GC'  : (0,0),
        'WL'  : (2,2),
        'GGL' : (0,2)}

    return coupling_matrix(bin_scheme, mask, wkspce_name, *spin_probe_dic[probe])

def coupling_matrix(bin_scheme, mask, wkspce_name, spin1, spin2):
    """
    Compute the coupling matrix for spherical harmonic modes using a binning scheme and mask.

    Parameters
    ----------
    bin_scheme : nmt_bins
        A binning scheme object defining the bins for the coupling matrix.
    mask : nmt_field
        A mask object defining the regions of the sky to include in the computation.
    wkspce_name : str
        The filename for storing or retrieving the computed workspace.
    spin1 : int
        Spin of the first field.
    spin2 : int
        Spin of the second field.

    Returns
    -------
    nmt_workspace
        Workspace object containing the computed coupling matrix.
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

#---- Covariance ----#

def decoupled_covariance(Cl, keys, wa, wb, cov_w, nbl):
    """
    Compute the decoupled covariance matrix using Gaussian covariance formalism.

    Parameters
    ----------
    Cl : dict
        Dictionary containing power spectra.
    keys : list
        List of correlation keys.
    wa : nmt_workspace
        Workspace object for one of the correlations.
    wb : nmt_workspace
        Workspace object for another correlation.
    cov_w : object
        Covariance workspace object.
    nbl : int
        Number of bins.

    Returns
    -------
    covmat : ndarray
        Decoupled covariance matrix.
    """
    ncl = len(keys)
    covmat = np.zeros((ncl*nbl, ncl*nbl))

    for (idx1, key1), (idx2, key2) in it.combinations_with_replacement(enumerate(keys), 2):
        print(key1, key2, flush=True)
        probeA, probeB = key1.split('-')
        probeC, probeD = key2.split('-')

        covmat[idx1*nbl:(idx1+1)*nbl, idx2*nbl:(idx2+1)*nbl] =\
            nmt.gaussian_covariance(cov_w, 0, 0, 0, 0,
                                    [Cl['-'.join([probeA, probeC])]],
                                    [Cl['-'.join([probeB, probeC])]],
                                    [Cl['-'.join([probeA, probeD])]],
                                    [Cl['-'.join([probeB, probeD])]],
                                    wa, wb=wb, coupled=False)
        covmat[idx2*nbl:(idx2+1)*nbl, idx1*nbl:(idx1+1)*nbl] =\
            covmat[idx1*nbl:(idx1+1)*nbl, idx2*nbl:(idx2+1)*nbl]

    return covmat

def coupled_covariance(Cl, keys, wa, wb, cov_w, nbl, weights=None):
    """
    Compute the coupled covariance matrix using Gaussian covariance formalism.

    Parameters
    ----------
    Cl : dict
        Dictionary containing power spectra.
    keys : list
        List of correlation keys.
    wa : nmt_workspace
        Workspace object for one of the correlations.
    wb : nmt_workspace
        Workspace object for another correlation.
    cov_w : object
        Covariance workspace object.
    nbl : int
        Number of bins.
    weights : ndarray, optional
        Weights for covariance binning.

    Returns
    -------
    covmat : ndarray
        Coupled covariance matrix.
    """
    assert wa.wsp.lmax == wb.wsp.lmax, 'The lmax from the two workspace is different.'

    ncl = len(keys)
    nell = wa.wsp.lmax+1
    covmat = np.zeros((ncl*nbl, ncl*nbl))

    for (idx1, key1), (idx2, key2) in it.combinations_with_replacement(enumerate(keys), 2):
        print(key1, key2, flush=True)
        probeA, probeB = key1.split('-')
        probeC, probeD = key2.split('-')

        unbinned_block =\
            nmt.gaussian_covariance(cov_w, 0, 0, 0, 0,
                                    [Cl['-'.join([probeA, probeC])]],
                                    [Cl['-'.join([probeB, probeC])]],
                                    [Cl['-'.join([probeA, probeD])]],
                                    [Cl['-'.join([probeB, probeD])]],
                                    wa, wb=wb, coupled=True)

        # bin each block
        covmat[idx1*nbl:(idx1+1)*nbl, idx2*nbl:(idx2+1)*nbl] =\
            covariance_binning_sum(unbinned_block, wa, weights=None)
        covmat[idx2*nbl:(idx2+1)*nbl, idx1*nbl:(idx1+1)*nbl] =\
            covmat[idx1*nbl:(idx1+1)*nbl, idx2*nbl:(idx2+1)*nbl]

    return covmat

def covariance_binning_sum(block_unbinned, workspace, weights=None):
    """
    Bin an unbinned covariance matrix using the provided workspace.

    Parameters
    ----------
    block_unbinned : ndarray
        Unbinned covariance matrix.
    workspace : nmt_workspace
        Workspace object for binning.
    weights : ndarray, optional
        Weights for binning.

    Returns
    -------
    binned_block : ndarray
        Binned covariance matrix.
    """
    if weights != None:
        raise ValueError('Weights are not yet implemented in the covariance binning !')

    nbins = workspace.get_bandpower_windows().shape[1]
    binning = workspace.wsp.bin
    binned_block = np.zeros((nbins,nbins))

    for bin_i in range(nbins):
        for bin_j in range(bin_i, nbins):
            # Select multipoles in the bins
            nell_in_bin_i = nmt.nmtlib.get_nell(binning, bin_i)
            nell_in_bin_j = nmt.nmtlib.get_nell(binning, bin_j)
            ells_i = nmt.nmtlib.get_ell_list(binning, bin_i, nell_in_bin_i)
            ells_j = nmt.nmtlib.get_ell_list(binning, bin_j, nell_in_bin_j)

            # Compute covariance sums
            sum_cov = 0
            for ell in ells_i:
                for ell_prime in ells_j:
                    sum_cov += block_unbinned[ell, ell_prime]

            # Normalize by the number of pairs of multipoles in the bins
            num_pairs = len(ells_i) * len(ells_j)
            binned_block[bin_i, bin_j] = sum_cov/num_pairs

    binned_block = binned_block + binned_block.T - np.diag(binned_block.diagonal())

    return binned_block

#---- Utils ----#
def probe_ref_mapping(probe):
    """
    Map probe types to reference names.

    Parameters
    ----------
    probe : str
        Probe type ('GC', 'WL', or 'GGL').

    Returns
    -------
    mapping : str
        Reference name corresponding to the probe type.
    """
    mapping = {
        'GC' : 'galaxy_cl',
        'WL' : 'shear_cl',
        'GGL' : 'galaxy_shear_cl'}

    return mapping[probe]

def nofz_to_euclidSGS(
    path: str | PathLike[str],
    z: ArrayLike,
    nz: ArrayLike,
    *,
    weight_method: str = "NO_WEIGHT",
    bin_type: str = "TOM_BIN",
    hist: bool = False,
) -> None:
    """
    This function is copied from https://github.com/euclidlib/euclidlib/blob/main/euclidlib/photo/_phz.py. Author: Nicolas Tessore 

    Write n(z) data in Euclid SGS format.  Supports both distributions
    (when *hist* is false, the default) and histograms (when *hist* is
    true).
    """

    z = np.asanyarray(z)
    nz = np.asanyarray(nz)

    if z.ndim != 1:
        raise ValueError("z array must be 1D")
    if nz.ndim == 0:
        raise ValueError("nz array must be at least 1D")
    if not hist and z.shape[-1] == nz.shape[-1]:
        pass
    elif hist and z.shape[-1] == nz.shape[-1] + 1:
        pass
    else:
        raise ValueError("shape mismatch between redshifts and values")

    # PHZ uses a fixed binning scheme with z bins in [0, 6] and dz=0.002
    zbinedges = np.linspace(0.0, 6.0, 3001)

    # turn nz into a 2D array with NBIN rows
    nz = nz.reshape(-1, nz.shape[-1])
    nbin = nz.shape[0]

    # create the output data in the correct format
    out = np.empty(
        nbin,
        dtype=[
            ("BIN_ID", ">i4"),
            ("MEAN_REDSHIFT", ">f4"),
            ("N_Z", ">f4", (3000,)),
        ],
    )

    # set increasing bin IDs
    out["BIN_ID"] = np.arange(1, nbin + 1)

    # convert every nz into the PHZ format
    if hist:
        # rebin the histogram as necessary

        # shorthand for the left and right z boundaries, respectively
        zl, zr = z[:-1], z[1:]

        # compute the mean redshifts
        out["MEAN_REDSHIFT"] = np.sum((zl + zr) / 2 * nz, axis=-1) / np.sum(nz, axis=-1)

        # compute resummed bin counts
        for j, (z1, z2) in enumerate(zip(zbinedges, zbinedges[1:])):
            frac = (np.clip(z2, zl, zr) - np.clip(z1, zl, zr)) / (zr - zl)
            out["N_Z"][:, j] = np.dot(nz, frac)
    else:
        # integrate the n(z) over each histogram bin

        # compute mean redshifts
        out["MEAN_REDSHIFT"] = trapezoid(z * nz, z, axis=-1) / trapezoid(nz, z, axis=-1)

        # compute the combined set of z grid points from data and binning
        zp = np.union1d(z, zbinedges)

        # integrate over each bin
        for i in range(nbin):
            # interpolate dndz onto the unified grid
            nzp = np.interp(zp, z, nz[i], left=0.0, right=0.0)

            # integrate the distribution over each bin
            for j, (z1, z2) in enumerate(zip(zbinedges, zbinedges[1:])):
                sel = (z1 <= zp) & (zp <= z2)
                out["N_Z"][i, j] = trapezoid(nzp[sel], zp[sel])

    # metadata
    header = {
        "WEIGHT_METHOD": weight_method,
        "BIN_TYPE": bin_type,
        "NBIN": nbin,
    }

    # write output data to FITS
    with fitsio.FITS(path, "rw", clobber=True) as fits:
        fits.write(out, extname="BIN_INFO", header=header)



def cells_to_euclidSGS(dic, bnmt, zbins, probes, cross, outname):

    """
    Convert Cells to Euclid SGS format.

    Parameters
    ----------
        Same as save_twopoint.py 
    """

    # format ell binning (we assume same ell-binning for the moment)
    ell = bnmt.get_effective_ells()
    nell = ell.size
    ell_max = np.zeros((nell))
    ell_min = np.zeros((nell))
    for i in range(bnmt.get_effective_ells().size):
        ell_max[i] = bnmt.get_ell_max(i)
        ell_min[i] = bnmt.get_ell_min(i)

    # Create FITS object
    hdr = fits.Header()
    hdr['COMMENT'] = " This is a FITS file in euclidlib format"
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([primary_hdu])

    # For each probe and  bin pair append a HDUTable
    for p in probes :
        if (p == 'GC'):
            tablename = 'POS-POS-'
            array = np.zeros((len(dic['ell']), ))
        elif (p == 'WL'):
            tablename = 'SHE-SHE-'
            array = np.zeros((len(dic['ell']), 2))
        else:
            tablename = 'POS-SHE-'
            array = np.zeros((len(dic['ell']), 2))

        probe_iter = get_iter(p, cross, zbins)
        print('----- :', array.shape, p)
        for pi, pj in probe_iter:
            if (p == 'GC'):
                array = dic['{}-{}'.format(pi,pj)]
                c1 = fits.Column(name='ARRAY', array=array, format='D')
                ellaxis = '(0,)'
            else:
                array[:,0] = dic['{}-{}'.format(pi,pj)]
                c1 = fits.Column(name='ARRAY', array=array, format='2D')
                ellaxis = '(1,)'
            c2 = fits.Column(name='ELL', array=ell, format='D')
            c3 = fits.Column(name='LOWER', array=ell_min, format='D')
            c4 = fits.Column(name='UPPER', array=ell_max, format='D')
            c5 = fits.Column(name='WEIGHT', array=np.ones(len(ell)), format='D')

            table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5])
            table_hdu.name = tablename +  pi[1] + '-' + pj[1]
            table_hdu.header['ELLAXIS'] = ellaxis
            hdul.append(table_hdu)

    hdul.writeto(outname, overwrite=True)
    hdul.close()

def mixmat_to_euclidSGS(dic, bnmt, zbins, probes, cross, outname):
    """
    Convert mixing matrices to Euclid SGS format. We're assuming 1 mixmat for each probe but constant in redshift bins.
    Euclid expects one for each probe and bin pair.

    Parameters
    ----------
        Same as save_twopoint.py 
    """

    # format ell binning (we assume same ell-binning for the moment)
    ell = bnmt.get_effective_ells()
    nell = ell.size
    ell_max = np.zeros((nell))
    ell_min = np.zeros((nell))
    for i in range(bnmt.get_effective_ells().size):
        ell_max[i] = bnmt.get_ell_max(i)
        ell_min[i] = bnmt.get_ell_min(i)

    # Create FITS object
    hdr = fits.Header()
    hdr['COMMENT'] = " This is a FITS file in euclidlib format"
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([primary_hdu])

    # For each probe and  bin pair append a HDUTable
    for p in probes :
        if (p == 'GC'):
            tablename = 'POS-POS-'
            ellaxis = '(0,)'
            array = np.zeros(dic[p].shape)
        elif (p == 'WL'):
            tablename = 'SHE-SHE-'
            ellaxis = '(1,)'
            array = np.zeros(dic[p].shape)
        else:
            tablename = 'POS-SHE-'
            ellaxis = '(1,)'
            array = np.zeros(dic[p].shape)

        probe_iter = get_iter(p, cross, zbins)

        for pi, pj in probe_iter:
            c1 = fits.Column(name='ARRAY', array=array, format=str(dic[p].shape[1])+'D')
            c2 = fits.Column(name='ELL', array=ell, format='D')
            c3 = fits.Column(name='LOWER', array=ell_min, format='D')
            c4 = fits.Column(name='UPPER', array=ell_max, format='D')
            c5 = fits.Column(name='WEIGHT', array=np.ones(len(ell)), format='D')

            table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5])
            table_hdu.name = tablename +  pi[1] + '-' + pj[1]
            table_hdu.header['ELLAXIS'] = ellaxis
            hdul.append(table_hdu)

    hdul.writeto(outname, overwrite=True)
    hdul.close()

def save_euclidlib(cls_dic, w_dic, bnmt, nofz_dic, zbins, probes, cross, outname):
    """
    Save nofz's, angular spectra and mixmat to a FITS file in Euclid SGS format.
    
    Parameters
    ----------
        Same as save_twopoint.py 
    """
    # 3/4 files will be created 
    outname_nz_lens = outname + '_nzlens.fits'
    outname_nz_source = outname + '_nzsource.fits'
    outname_cells = outname + '_cells.fits'
    outname_mixmat = outname + '_mixmat.fits'

    # FITSFILEs 1 and 2: nofz for lenses and sources (we use Nicolas's function)
    nofz_to_euclidSGS(outname_nz_lens, nofz_dic['lens'][0], nofz_dic['lens'][1:])
    nofz_to_euclidSGS(outname_nz_source, nofz_dic['source'][0], nofz_dic['source'][1:])

    # FITSFILEs 3 and 4: cells and mixing matrix
    cells_to_euclidSGS(cls_dic, bnmt, zbins, probes, cross, outname_cells)
    mixmat_to_euclidSGS(w_dic,  bnmt, zbins, probes, cross, outname_mixmat)