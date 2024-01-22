import numpy as np

def numpy_save(outname, cls):
    np.save('{}.npy'.format(outname), cls)