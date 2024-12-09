import numpy as np
import itertools as it

def cov_to_corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def pos_def(cov, which):
    try:
        np.linalg.cholesky(cov)
        print(which, ' is positive definite')
    except:
        print(which, ' is NOT positive definite')

def is_symmetric(cov):
    test = (cov == cov.T)
    if np.where(test==False)[0].size == 0:
        print ('is symmetric')
    else :
        print('not symmetric')


def errs_from_cov(cov, probe, nzbins, cross):
    """
    Compute errors from the covariance matrix.

    Parameters
    ----------
    cov : array-like
        Covariance matrix.
    probe : str
        Probe type ('GC', 'WL', or 'GGL').
    nzbins : int
        Number of redshift bins.
    cross : bool
        Flag indicating whether to compute cross-correlations.

    Returns
    -------
    err_dic : dict
        Dictionary containing error arrays for each bin pair.

    Notes
    -----
    - If `probe` is 'GGL', the covariance matrix is assumed to be asymmetric.
    - The number of pairs (`npairs`) is determined based on the probe type and cross-correlation flag.
    - The function iterates over pairs of redshift bins and reshapes the error array accordingly.
    - The errors are stored in a dictionary with keys representing the bin pair labels.

    Examples
    --------
    >>> cov = np.random.rand(100, 100)  # Example covariance matrix
    >>> probe = 'GC'
    >>> nzbins = 5
    >>> cross = True
    >>> errs_from_cov(cov, probe, nzbins, cross)
    {'0-0': array([...]), '0-1': array([...]), ...}
    """
    symmetric = True
    if probe == 'GGL':
        symmetric = False

    if not cross:
        npairs = nzbins
    elif symmetric and cross:
        npairs = int((nzbins * (nzbins - 1) / 2 + nzbins))
    elif not symmetric and cross:
        npairs = int(nzbins * nzbins)

    cross_list = []
    if cross:
        cross_list = [probe]

    iterator = get_iter(probe, cross_list, np.arange(nzbins))

    nell = int(np.sqrt(cov.size) / npairs)
    print(nell, ' nell')
    err_2d = np.reshape(np.sqrt(np.diag(cov)), (npairs, nell))
    err_dic = {}
    idx = 0
    for (i, j) in iterator:
        err_dic['{}-{}'.format(i, j)] = err_2d[idx]
        idx += 1
    return err_dic

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
    keymap = {'GC': ['D{}'.format(i) for i in zbins],
              'WL': ['G{}'.format(i) for i in zbins],
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


