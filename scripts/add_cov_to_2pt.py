import twopoint
import numpy as np
from astropy.io import fits

# fsky = 0.125

twopoint_file = '/Users/sgouyoub/Documents/work/euclid/3x2/data/measurement/FS2_octant_WL_6EP_noIA-nomag-nored-nowh_sharpcut_shapenoise_decoupled/FS2_octant_WL_6EP_noIA-nomag-nored-nowh_sharpcut_shapenoise_decoupled_Cls_NS1024_LBINlog_LMIN10_LMAX1500_NELL32.fits'
data = twopoint.TwoPointFile.from_fits(twopoint_file, covmat_name=None)

cov_file = '/Users/sgouyoub/Documents/work/euclid/3x2/data/covariance/FS2_octant_WL_6EP_noIA-nomag-nored-nowh_sharpcut_covariance_nmt_decoupled_NS1024_LBINlog_LMIN10_LMAX1500_NELL32.npy'
cov = np.load(cov_file)
# hdul = fits.open('/Users/sgouyoub/Documents/work/soft/cosmosis-standard-library/working_dir/output/30deg_nlbias/simulation/30deg_6EP_WL_fsky_SDV/30deg_6EP_WL_fsky_SDV.fits')
# cov = hdul['COVMAT'].data
# fsky = 0.06712301572163899
# cov *= fsky**2

names = [s.name for s in data.spectra]
lengths = [len(s) for s in data.spectra]

n = sum(lengths)
assert cov.shape == (n,n)

data.covmat_info = twopoint.CovarianceMatrixInfo("COVMAT", names, lengths, cov)
# twopoint_file = '/Users/sgouyoub/Documents/work/euclid/nlbias/data/measurement/FS2_30deg_WL_6EP_noIA-nomag-nored-nowh_sharpcut_shapenoise_GaussSimCov_Cls_NS1024_LBINlog_LMIN10_LMAX1500_NELL32.fits'
data.to_fits(twopoint_file, clobber=True)