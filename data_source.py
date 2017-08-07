import os
import random

import numpy as np
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

from nnabla.utils import data_source


class DataSource(data_source.DataSource):
    def __init__(self, paths, root, shuffle=None, rng=None):
        super(DataSource, self).__init__(shuffle=shuffle, rng=rng)
        super(DataSource, self).reset()
        if shuffle:
            self._order = list(
                np.random.permutation(list(range(self._size))))
        else:
            self._order = list(range(self._size))
        self.paths = paths
        self.root = root
        self._size = len(paths)
        self._variables = ['x' + str(x)
                           for x in range(len(self._get_data(0)))]

    def _get_data(self, position):
        path = self.paths[position]
        f = Image.open(os.path.join(self.root, path))
        # f = f.resize((109, 89), Image.ANTIALIAS)
        try:
            image = np.asarray(f, dtype=np.float32)
        finally:
            # Only pillow >= 3.0 has 'close' method
            if hasattr(f, 'close'):
                f.close()
        image = image.transpose(2, 0, 1)
        _, h, w = image.shape
        top = random.randint(0, h - 128 - 1)
        left = random.randint(0, w - 128 - 1)
        if random.randint(0, 1):
            image = image[:, :, ::-1]

        bottom = top + 128
        right = left + 128

        image = image[:, top:bottom, left:right]
        image *= 2 / 255
        image -= 1
        return np.expand_dims(image, 0)
