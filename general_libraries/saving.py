import numpy as np
import itertools as it

def save_cloe_cls(data, nbins, probe, cross, outname):

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
        if probe == 'GGL':
            dic['{}{}-{}{}'.format(ki, i-1, kj, j-1)] = vec[:,c]
        else:
            dic['{}{}-{}{}'.format(ki, i-1, kj, j-1)] = vec[:,c]
        c+=1
    dic['ell'] = ell

    np.save(outname, dic)