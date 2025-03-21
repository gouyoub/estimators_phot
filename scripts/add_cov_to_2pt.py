import twopoint
import numpy as np
from astropy.io import fits

# fsky = 0.125

twopoint_file = '/Users/sgouyoub/Documents/work/euclid/nlbias/data/SDV/30deg_6EP_3x2_nonlinbias_coupled_SDV_bfnl3_noisy_4.fits'
data = twopoint.TwoPointFile.from_fits(twopoint_file, covmat_name=None)
# data = twopoint.TwoPointFile.from_fits(twopoint_file)

cov_file = '/Users/sgouyoub/Documents/work/euclid/3x2/data/covariance/FS2_30deg_3x2_6EP_noIA-nomag-nored-nowh_sharpcut_covariance_b1fromfit_nmt_coupled_NS1024_LBINlog_LMIN10_LMAX1500_NELL32.npy'
cov = np.load(cov_file)
# hdul = fits.open('/Users/sgouyoub/Documents/work/soft/cosmosis-standard-library/working_dir/output/30deg/simulation/30deg_6EP_3x2_linbias_fsky_4cov_SDV/30deg_6EP_3x2_linbias_fsky_4cov_SDV.fits')
# cov = hdul['COVMAT'].data

names = [s.name for s in data.spectra]
lengths = [len(s) for s in data.spectra]

n = sum(lengths)
assert cov.shape == (n,n)

data.covmat_info = twopoint.CovarianceMatrixInfo("COVMAT", names, lengths, cov)
data.to_fits(twopoint_file, clobber=True)