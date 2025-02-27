from __future__ import print_function
import os.path as osp
import os
import numpy as np
import glob
from ..serialization import read_json


def _pluck(root):
    ret = []
    id_num = 0
    path = glob.glob(root + '/*/*/*.jpg')
    for index, (fname) in enumerate(path):
        pid = int(fname[-34:-31])
        color = int(fname[-30])
        type = int(fname[-28])
        pose = int(fname[-26])
        if pid > id_num:
            id_num = pid
        camid = int(fname[-18:-15])
        ret.append((fname, pid, pose, camid, color, type))

    return ret, id_num+1


class Dataset(object):
    def __init__(self, root):
        self.root = root
        self.train = []
        self.num_train_ids = 0

    def __len__(self):
        return

    def load(self, verbose=True):
        # fname, pid, camid, color, type
        self.train, self.num_train_ids = _pluck(self.root)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}".format(self.num_train_ids, len(self.train)))

