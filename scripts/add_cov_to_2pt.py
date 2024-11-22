import twopoint
import numpy as np
from astropy.io import fits

fsky = 0.125

twopoint_file = '/home/hidra2/gouyou/euclid/nl_bias_flagship/data/measurement/FS2_3x2_firstchain_decoupled_log/FS2_3x2_firstchain_decoupled_log_Cls_NS1024_LBINlog_LMIN10_LMAX2048_NELL32.fits'
data = twopoint.TwoPointFile.from_fits(twopoint_file, covmat_name=None)
# data = twopoint.TwoPointFile.from_fits(twopoint_file)

cov_file = '/home/hidra2/gouyou/euclid/nl_bias_flagship/data/covariance/FS2_3x2_firstchain_partialsky_6ED_NS1024_LBINlog_her_LMIN10_NELL32.npy'
cov = np.load(cov_file)
# hdul = fits.open('/home/hidra2/gouyou/euclid/nl_bias_flagship/data/measurement/FS2_3x2_firstchain_pskycov_Cls_NS1024_LBINlin_LMIN10_BW50.fits')
# cov = np.diag(np.diag(hdul['COVMAT'].data))

names = [s.name for s in data.spectra]
lengths = [len(s) for s in data.spectra]

n = sum(lengths)
assert cov.shape == (n,n)

data.covmat_info = twopoint.CovarianceMatrixInfo("COVMAT", names, lengths, cov)
data.to_fits(twopoint_file, clobber=True)