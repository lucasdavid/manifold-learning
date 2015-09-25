import numpy as np

from . import errors


class Retriever(object):
    """Retriever for machine learning data set files.

    Usage:
        r = Retriever('path/to/file')
        data, _ = r.load().retrieve()

        # if you know which column contains the target feature:
        data, target = r.split_target(target_column=-1).retrieve()
    """

    def __init__(self, file, target_column=-1, delimiter=None):
        """Initiates a Retriever pointing out to a data-set file.

        :param file: the file which contains the data-set.
        :param target_column: the column which contains the target feature, if any.
        :param delimiter: the character used to separate the samples' features.
                          If none specified, /s and/or /t are considered.
        """
        self.target_column = target_column
        self.delimiter = delimiter

        self._file = file
        self._data = self._target = self._features_map = None

    @property
    def loaded(self):
        """Checks if any data-set has been loaded.
        """
        return self._data is not None

    @property
    def features_map(self):
        self.load()

        return self._features_map

    @property
    def features_count(self):
        """Retrieves the number of features that the loaded data set has.
        """
        self.load()

        return len(self._data.shape[0])

    def load(self):
        """Loads data set from file.

        :return: :raise errors.RetrieverError: if any errors were raised when manipulating the file.
        """
        if not self.loaded:
            self._features_map = {}
            self._data = []

            try:
                with open(self._file) as f:
                    for line in f.readlines():
                        row = []

                        for i, word in enumerate(line.split(self.delimiter)):
                            try:
                                word = float(word)
                            except ValueError:
                                # Failed to parse float, as this column represents a nominal feature.
                                if i not in self._features_map:
                                    self._features_map[i] = {}

                                if word not in self._features_map[i]:
                                    self._features_map[i][word] = len(self._features_map[i])

                                # Replace word by its enumeration.
                                word = self._features_map[i][word]

                            row.append(word)

                        self._data.append(row)

            except IOError as error:
                # File doesn't exist; user doesn't have necessary permissions or data parsing failed.
                raise errors.RetrieverError(error)

            self._data = np.array(self._data)

        return self

    def split_column(self, column):
        """Strip a column from the data. Then returns it.

        :param column: the column that will be removed from the data.
        """
        self.load()

        result = self._data[:, column]
        self._data = np.delete(self._data, column, 1)

        return result

    def split_target(self, target_column=None):
        """Split target column from data.

         Useful if one of the columns represent a target feature.

        :param target_column: the column that contains the target feature. When None, uses the value passed in the
        construction of the object. If None, considers the last column (-1).
        """
        if self._target is None:
            # If :_target, target has already been striped from data.
            self.load()

            if target_column is not None:
                # target_column informed. It could be different from -1,
                # which is the default target-column value.
                self.target_column = target_column

            # Effectively split target-column from data.
            self._target = self.split_column(self.target_column)

        return self

    def retrieve(self):
        self.load()

        return self._data, self._target
