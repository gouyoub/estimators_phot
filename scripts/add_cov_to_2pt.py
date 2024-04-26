import twopoint
import numpy as np

twopoint_file = '/home/hidra2/gouyou/euclid/nl_bias_flagship/data/measurement/FS2_GCauto_firstchain_Cls_NS1024_LMIN10_BW30.fits'
data = twopoint.TwoPointFile.from_fits(twopoint_file, covmat_name=None)

cov_file = '/home/hidra2/gouyou/euclid/nl_bias_flagship/data/covariance/FS2_GCauto_firstchain.npy'
cov = np.load(cov_file)

names = [s.name for s in data.spectra]
lengths = [len(s) for s in data.spectra]

n = sum(lengths)
assert cov.shape == (n,n)

data.covmat_info = twopoint.CovarianceMatrixInfo("COVMAT", names, lengths, cov)
data.to_fits(twopoint_file, clobber=True)