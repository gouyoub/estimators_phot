import numpy as np

def chi2(data, model, cov):
    delta = data-model
    psi = np.linalg.inv(cov)
    chi2 = delta@psi@delta
    return chi2
