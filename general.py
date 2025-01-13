import numpy as np
import itertools as it

def mysplit(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return head, tail

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

def pos_def(cov, which):
    try:
        np.linalg.cholesky(cov)
        print(which, ' is positive definite')
    except:
        print(which, ' is NOT positive definite')

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