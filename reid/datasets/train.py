from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class train(Dataset):

    def __init__(self, root):
        super(train, self).__init__(root)

        self.load()
