import numpy as np
from astropy.io import fits
from heracles.core import TocDict
import itertools as it


def load_it(obj):
    """
    Recursively loads a nested structure by converting string representations
    to their appropriate Python types.

    Parameters
    ----------
    obj : dict, list, str, int, float, bool, or None
        The object or structure to be loaded.

    Returns
    -------
    loaded_obj : dict, list, int, float, bool, or None
        The loaded object or structure with string representations converted to
        their appropriate Python types.

    Examples
    --------
    >>> input_obj = {'key1': '123', 'key2': ['None', '3.14', 'True']}
    >>> load_it(input_obj)
    {'key1': 123, 'key2': [None, 3.14, True]}

    Notes
    -----
    This function recursively iterates through a nested structure (dict or list)
    and converts string representations to their appropriate Python types.
    - Strings 'None' are converted to `None`.
    - Numeric strings are converted to integers or floats.
    - String representations of boolean values ('True' or 'False') are converted
      to corresponding boolean values.
    - Strings representing lists are converted to actual lists.

    """
    if isinstance(obj, dict):
        return {k: load_it(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [load_it(elem) for elem in obj]
    if isinstance(obj, str):
        if obj == 'None':
            return None
        if obj.startswith('[') and obj.endswith(']'):
            return [load_it(elem.strip()) for elem in obj[1:-1].split(',')]
        if obj.isnumeric():
            return int(obj)
        if obj.replace('.', '', 1).isnumeric():
            return float(obj)
        if obj.upper() in ('TRUE', 'FALSE', 'T', 'F'):
            return obj.upper() in ('TRUE', 'T')

    return obj

def get_cosmosis_autoCl(ref,ell,nbins):
    Cl = np.zeros((nbins, int(ell.size)))
    for i in range(nbins):
        Cl[i] = np.loadtxt(ref+'bin_'+str(i+1)+'_'+str(i+1)+'.txt')
    return Cl

def get_cosmosis_Cl(ref, nbins, probe, cross):

    prob_ref = {'GC' : 'galaxy_cl', 'WL' : 'shear_cl', 'GGL': 'galaxy_shear_cl'}
    ref += prob_ref[probe]+'/'

    indices     = [i+1 for i in range(nbins)]

    cross_selec = {True  : it.combinations_with_replacement(indices, 2),
                   False : zip(indices, indices)}
    iter_probe  = {'GC'  : cross_selec[cross],
                  'WL'  : cross_selec[cross],
                  'GGL' : it.product(indices, indices)}

    keymap   = {'GC'  : ('D','D'),
                'WL'  : ('G','G'),
                'GGL' : ('D','G')}

    dic = {}
    keys = []
    for i,j in iter_probe[probe]:
        ki, kj = keymap[probe]
        if probe == 'GGL':
            dic['{}{}-{}{}'.format(ki, i-1, kj, j-1)] = np.loadtxt(ref+'bin_{}_{}.txt'.format(i,j))
        else:
            dic['{}{}-{}{}'.format(ki, i-1, kj, j-1)] = np.loadtxt(ref+'bin_{}_{}.txt'.format(j,i))
        keys.append('{}{}-{}{}'.format(ki, i-1, kj, j-1))

    dic['ell'] = np.loadtxt(ref+'ell.txt')

    return dic, keys

def get_cls_nmt_gc(fitsfile):
    hdul = fits.open(fitsfile)
    n_entry = hdul[1].header['TFIELDS']
    cls = {}
    for i in range(n_entry):
        key = hdul[1].header['TTYPE'+str(i+1)]
        cls[key] = hdul[1].data[key]
    return cls

def get_cls_nmt_txt(txtfile, probe):
    spin = {'wl':4, 'ggl':2, 'gc':1}
    modes = {'wl': {0:'EE', 1:'EB', 2:'BE', 3:'BB'},
             'ggl': {0:'PE', 1:'PB'},
             'gc': {0:'PP'}}
    cls_temp = np.loadtxt(txtfile)
    nbins = int(cls_temp.shape[0]/spin[probe])
    cls = {}
    for i in range(nbins):
        for j in range(spin[probe]):
            key = modes[probe][j]+'_'+str(i+1)
            cls[key] = cls_temp[i*spin[probe]+j]
    return cls

def get_cloe_cls(filename, nbins, probe, cross):

    data = np.loadtxt(filename)
    ell = data[:,0]
    vec = data[:,1:]

    indices     = [i+1 for i in range(nbins)]

    cross_selec = {True  : it.combinations_with_replacement(indices, 2),
                   False : zip(indices, indices)}
    iter_probe  = {'GC'  : cross_selec[cross],
                  'WL'  : cross_selec[cross],
                  'GGL' : it.product(indices, indices)}

    keymap   = {'GC'  : ('D','D'),
                'WL'  : ('G','G'),
                'GGL' : ('D','G')}

    dic = {}
    c=0
    for i,j in iter_probe[probe]:
        ki, kj = keymap[probe]
        dic['{}{}-{}{}'.format(ki, i-1, kj, j-1)] = vec[:,c]
        c+=1
    dic['ell'] = ell

    return dic

def get_cloe_err(filename, nzbins, nell, probe):
    covmat = np.load(filename)
    nelem = covmat.shape[0]

    npairs = {'GC' : int(nzbins*(nzbins-1)/2 + nzbins),
              'WL' : int(nzbins*(nzbins-1)/2 + nzbins),
              'GGL': int(nzbins**2)}

    selec_range = {'GC' : np.arange(nelem)[npairs['WL']*nell + npairs['GGL']*nell:],
                   'WL' : np.arange(nelem)[:npairs['WL']*nell],
                   'GGL': np.arange(nelem)[npairs['WL']*nell:npairs['GGL']*nell]}

    var_vec = np.diag(covmat)[selec_range[probe]]
    var_reshape = np.reshape(var_vec, (nell, npairs[probe]))

    indices     = [i+1 for i in range(nzbins)]

    iter_probe  = {'GC'  : it.combinations_with_replacement(indices, 2),
                  'WL'  : it.combinations_with_replacement(indices, 2),
                  'GGL' : it.product(indices, indices)}

    keymap   = {'GC'  : ('D','D'),
                'WL'  : ('G','G'),
                'GGL' : ('D','G')}

    err_dic = {}
    c=0
    for i,j in iter_probe[probe]:
        ki, kj = keymap[probe]
        if probe == 'GGL':
            err_dic['{}{}-{}{}'.format(ki, i-1, kj, j-1)] = np.sqrt(var_reshape[:,c])
        else:
            err_dic['{}{}-{}{}'.format(ki, i-1, kj, j-1)] = np.sqrt(var_reshape[:,c])
        c+=1
    return err_dic

def trans_cldicCS_clarrayDAV(cl_dic, probe):

    # Number of ells and pairs
    nbl = cl_dic['ell'].size
    ncl = len(cl_dic.keys())-1

    # Number of redshift bins
    if probe == 'GGL':
        nbz = int(np.sqrt(ncl))
    else :
        nbz = int(1/2*(np.sqrt(8*ncl+1)-1))

    #Initialise the Cl array
    cl_array = np.zeros((nbl, nbz, nbz))

    # Define iteration schemes for different probes
    indices = range(nbz)
    iter_probe = {'GC': it.combinations_with_replacement(indices, 2),
                  'WL': it.combinations_with_replacement(indices, 2),
                  'GGL': it.product(indices, repeat=2)}

    # Mapping between probe type and key format
    keymap = {'GC': ('D', 'D'),
              'WL': ('G', 'G'),
              'GGL': ('D', 'G')}

    # Loop to fill the Cl array
    for i, j in iter_probe[probe]:
        cl_array[:, i, j] = cl_dic['{}{}-{}{}'.format(keymap[probe][0], i, keymap[probe][1], j)]
        if (probe == 'WL' or probe == 'GC'):
            cl_array[:, j, i] = cl_dic['{}{}-{}{}'.format(keymap[probe][0], i, keymap[probe][1], j)]

    return cl_array

def get_cls_2pt(ref, nbins, probe, cross, cl_name='namaster'):

    if cl_name == 'cosmosis' :
        prob_ref = {'GC' : 'galaxy_cl', 'WL' : 'shear_cl', 'GGL': 'galaxy_shear_cl'}
    elif cl_name == 'namaster' :
        prob_ref = {'GC' : 'cellGC', 'WL' : 'cellWL', 'GGL': 'cellGGL'}
    else :
        raise ValueError('cl_name needs to be cosmosis or namaster')

    indices     = [i+1 for i in range(nbins)]

    cross_selec = {True  : it.combinations_with_replacement(indices, 2),
                   False : zip(indices, indices)}
    iter_probe  = {'GC'  : cross_selec[cross],
                  'WL'  : cross_selec[cross],
                  'GGL' : it.product(indices, indices)}

    keymap   = {'GC'  : ('D','D'),
                'WL'  : ('G','G'),
                'GGL' : ('D','G')}

    hdul = fits.open(ref)
    nell = 0
    for i in hdul[prob_ref[probe]].data['ANGBIN']:
        if hdul[prob_ref[probe]].data['ANGBIN'][i+1] == 0:
            nell = hdul[prob_ref[probe]].data['ANGBIN'][i]+1

    dic = {}
    keys = []
    c=0
    for i,j in iter_probe[probe]:
        ki, kj = keymap[probe]
        dic['{}{}-{}{}'.format(ki, i-1, kj, j-1)] = hdul[prob_ref[probe]].data['VALUE'][c*nell:(c+1)*nell]
        keys.append('{}{}-{}{}'.format(ki, i-1, kj, j-1))
        c+=1

    dic['ell'] = hdul[prob_ref[probe]].data['ANG'][:nell]

    return dic, keys

def get_cls_heracles(fitsfile):
    hdul = fits.open(fitsfile)
    toc = hdul[1].data
    cls = {}
    for i in range(toc.size):
        cls[toc[i][1], toc[i][2], toc[i][3], toc[i][4]] = hdul[i+2].data
    return TocDict(cls)
