import twopoint
import numpy as np
from astropy.io import fits

fsky = 0.125

twopoint_file = '/home/hidra2/gouyou/euclid/nl_bias_flagship/data/measurement/FS2_GCauto_firstchain_NmtCov_Cls_NS2048_LBINlin_LMIN10_BW50.fits'
data = twopoint.TwoPointFile.from_fits(twopoint_file, covmat_name=None)
# data = twopoint.TwoPointFile.from_fits(twopoint_file)

cov_file = '/home/hidra2/gouyou/euclid/nl_bias_flagship/data/covariance/FS2_GCauto_firstchain_6ED_NS2048_LBINlin_LMIN10_BW50.npy'
cov = np.load(cov_file)
# hdul = fits.open('/home/hidra2/gouyou/cosmosis-standard-library/working_dir/output/fs2_nlbias/simulation/SDV_3x2_fs2_6ED_linbias_covmat2/SDV_3x2_fs2_6ED_linbias_covmat2.fits')
# cov = hdul['COVMAT'].data

names = [s.name for s in data.spectra]
lengths = [len(s) for s in data.spectra]

n = sum(lengths)
assert cov.shape == (n,n)

data.covmat_info = twopoint.CovarianceMatrixInfo("COVMAT", names, lengths, cov)
data.to_fits(twopoint_file, clobber=True)