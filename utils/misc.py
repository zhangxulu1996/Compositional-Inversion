import os
import numpy as np
import random
import torch


def check_mk_file_dir(file_name):
    check_mkdir(file_name[:file_name.rindex("/")])
    
def check_mkdir(dir_name):
    """
    check if the folder exists, if not exists, the func will create the new named folder.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def set_seed(seed = 8888):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


class Dict2Class(object):
    def __init__(self, mydict):
        self.dict = mydict
        for key in mydict.keys():
            setattr(self, key, mydict[key])

    def get_dict(self):
        return self.dict
