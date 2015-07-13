import numpy as np

from . import errors


class DataSetRetriever(object):
    def __init__(self, file, target_column=-1):
        self.target_column = target_column

        self._file = file
        self._data = self._target = None

    @property
    def loaded(self):
        """Checks if any data-set has been loaded.
        """
        return self._data is not None

    @property
    def features(self):
        """Retrieves the number of features that the loaded data set has.
        """
        self.load()

        return len(self._data[0])

    def load(self):
        """Loads data set from file.

        :return: :raise errors.RetrieverError: if any errors were raised when manipulating the file.
        """
        if not self.loaded:
            data = []

            try:
                with open(self._file) as f:
                    raw_data = f.readlines()

                    for row in raw_data:
                        words = row.split()
                        data.append([float(e) for e in words])

            except (IOError, ValueError) as e:
                raise errors.RetrieverError(e)

            self._data = np.array(data)

        return self

    def split_target(self, target_column=None):
        """Strip a target column from the data.

         Useful if one of the columns represent a target feature.

        :param target_column: the column that contains the target feature. When None, uses the value passed in the
        construction of the object. If None, considers the last column (-1).
        """
        self.load()

        if target_column is not None:
            self.target_column = target_column

        self._target = self._data[:, self.target_column]
        self._data = np.delete(self._data, self.target_column, 1)

        return self

    def retrieve(self):
        return self._data, self._target
