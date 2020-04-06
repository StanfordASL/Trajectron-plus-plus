import numpy as np
import pandas as pd
from collections import Sequence, OrderedDict


class RingBuffer(Sequence):
    def __init__(self, capacity, dtype=float, allow_overwrite=True):
        """
        Create a new ring buffer with the given capacity and element type.
        Code copy-pasted from: https://github.com/eric-wieser/numpy_ringbuffer

        Parameters
        ----------
        capacity: int
            The maximum capacity of the ring buffer
        dtype: data-type, optional
            Desired type of buffer elements. Use a type like (float, 2) to
            produce a buffer with shape (N, 2)
        allow_overwrite: bool
            If false, throw an IndexError when trying to append to an already
            full buffer
        """
        self._arr = np.full(capacity, np.nan, dtype)
        self._left_index = 0
        self._right_index = 0
        self._capacity = capacity
        self._allow_overwrite = allow_overwrite

    def _unwrap(self):
        """ Copy the data from this buffer into unwrapped form """
        return np.concatenate((
            self._arr[self._left_index:min(self._right_index, self._capacity)],
            self._arr[:max(self._right_index - self._capacity, 0)]
        ))

    def _fix_indices(self):
        """
        Enforce our invariant that 0 <= self._left_index < self._capacity
        """
        if self._left_index >= self._capacity:
            self._left_index -= self._capacity
            self._right_index -= self._capacity
        elif self._left_index < 0:
            self._left_index += self._capacity
            self._right_index += self._capacity

    @property
    def is_full(self):
        """ True if there is no more space in the buffer """
        return len(self) == self._capacity

    # numpy compatibility
    def __array__(self):
        return self._unwrap()

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def shape(self):
        return (len(self),) + self._arr.shape[1:]

    # these mirror methods from deque
    @property
    def maxlen(self):
        return self._capacity

    def append(self, value):
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError('append to a full RingBuffer with overwrite disabled')
            elif not len(self):
                return
            else:
                self._left_index += 1

        self._arr[self._right_index % self._capacity] = value
        self._right_index += 1
        self._fix_indices()

    def appendleft(self, value):
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError('append to a full RingBuffer with overwrite disabled')
            elif not len(self):
                return
            else:
                self._right_index -= 1

        self._left_index -= 1
        self._fix_indices()
        self._arr[self._left_index] = value

    def pop(self):
        if len(self) == 0:
            raise IndexError("pop from an empty RingBuffer")
        self._right_index -= 1
        self._fix_indices()
        res = self._arr[self._right_index % self._capacity]
        return res

    def popleft(self):
        if len(self) == 0:
            raise IndexError("pop from an empty RingBuffer")
        res = self._arr[self._left_index]
        self._left_index += 1
        self._fix_indices()
        return res

    def extend(self, values):
        lv = len(values)
        if len(self) + lv > self._capacity:
            if not self._allow_overwrite:
                raise IndexError('extend a RingBuffer such that it would overflow, with overwrite disabled')
            elif not len(self):
                return
        if lv >= self._capacity:
            # wipe the entire array! - this may not be threadsafe
            self._arr[...] = values[-self._capacity:]
            self._right_index = self._capacity
            self._left_index = 0
            return

        ri = self._right_index % self._capacity
        sl1 = np.s_[ri:min(ri + lv, self._capacity)]
        sl2 = np.s_[:max(ri + lv - self._capacity, 0)]
        self._arr[sl1] = values[:sl1.stop - sl1.start]
        self._arr[sl2] = values[sl1.stop - sl1.start:]
        self._right_index += lv

        self._left_index = max(self._left_index, self._right_index - self._capacity)
        self._fix_indices()

    def extendleft(self, values):
        lv = len(values)
        if len(self) + lv > self._capacity:
            if not self._allow_overwrite:
                raise IndexError('extend a RingBuffer such that it would overflow, with overwrite disabled')
            elif not len(self):
                return
        if lv >= self._capacity:
            # wipe the entire array! - this may not be threadsafe
            self._arr[...] = values[:self._capacity]
            self._right_index = self._capacity
            self._left_index = 0
            return

        self._left_index -= lv
        self._fix_indices()
        li = self._left_index
        sl1 = np.s_[li:min(li + lv, self._capacity)]
        sl2 = np.s_[:max(li + lv - self._capacity, 0)]
        self._arr[sl1] = values[:sl1.stop - sl1.start]
        self._arr[sl2] = values[sl1.stop - sl1.start:]

        self._right_index = min(self._right_index, self._left_index + self._capacity)

    # implement Sequence methods
    def __len__(self):
        return self._right_index - self._left_index

    def __getitem__(self, item):
        # handle simple (b[1]) and basic (b[np.array([1, 2, 3])]) fancy indexing specially
        if not isinstance(item, tuple):
            item_arr = np.asarray(item)
            if issubclass(item_arr.dtype.type, np.integer):
                item_arr = (item_arr + self._left_index) % self._capacity
                return self._arr[item_arr]

        # for everything else, get it right at the expense of efficiency
        return self._unwrap()[item]

    def __iter__(self):
        # alarmingly, this is comparable in speed to using itertools.chain
        return iter(self._unwrap())

    # Everything else
    def __repr__(self):
        return '<RingBuffer of {!r}>'.format(np.asarray(self))


class DoubleHeaderNumpyArray(object):
    def __init__(self, data: np.ndarray, header: list):
        """
        Data Structure mirroring some functionality of double indexed pandas DataFrames.
        Indexing options are:
        [:, (header1, header2)]
        [:, [(header1, header2), (header1, header2)]]
        [:, {header1: [header21, header22]}]

        A SingleHeaderNumpyArray can is returned if an element of the first header is querried as attribut:
        doubleHeaderNumpyArray.position -> SingleHeaderNumpyArray

        :param data: The numpy array.
        :param header: The double header structure as list of tuples [(header11, header21), (header11, header22) ...]
        """
        self.data = data
        self.header = header
        self.double_header_lookup = OrderedDict()
        self.tree_header_lookup = OrderedDict()
        for i, header_item in enumerate(header):
            self.double_header_lookup[header_item] = i
            if header_item[0] not in self.tree_header_lookup:
                self.tree_header_lookup[header_item[0]] = dict()
            self.tree_header_lookup[header_item[0]][header_item[1]] = i

    def __mul__(self, other):
        return DoubleHeaderNumpyArray(self.data * other, self.header)

    def get_single_header_array(self, h1: str, rows=slice(None, None, None)):
        data_integer_indices = list()
        h2_list = list()
        for h2 in self.tree_header_lookup[h1]:
            data_integer_indices.append(self.tree_header_lookup[h1][h2])
            h2_list.append(h2)
        return SingleHeaderNumpyArray(self.data[rows, data_integer_indices], h2_list)

    def __getitem__(self, item):
        rows, columns = item
        data_integer_indices = list()
        if type(columns) is dict:
            for h1, h2s in columns.items():
                for h2 in h2s:
                    data_integer_indices.append(self.double_header_lookup[(h1, h2)])
            return self.data[rows, data_integer_indices]
        elif type(columns) is list:
            for column in columns:
                assert type(column) is tuple, "If Index is list it hast to be list of double header tuples."
                data_integer_indices.append(self.double_header_lookup[column])
            return self.data[rows, data_integer_indices]
        elif type(columns) is tuple:
            return self.data[rows, self.double_header_lookup[columns]]
        else:
            assert type(item) is str, "Index must be str, list of tuples or dict of tree structure."
            return self.get_single_header_array(item, rows=rows)

    def __getattr__(self, item):
        if not item.startswith('_'):
            if item in self.tree_header_lookup.keys():
                return self.get_single_header_array(item)
            else:
                try:
                    return self.data.__getattribute__(item)
                except AttributeError:
                    return super().__getattribute__(item)
        else:
            return super().__getattribute__(item)


class SingleHeaderNumpyArray(object):
    def __init__(self, data: np.ndarray, header: list):
        self.data = data
        self.header_lookup = OrderedDict({h: i for i, h in enumerate(header)})

    def __getitem__(self, item):
        rows, columns = item
        data_integer_indices = list()
        if type(columns) is list or type(columns) is tuple:
            for column in columns:
                data_integer_indices.append(self.header_lookup[column])
        else:
            data_integer_indices = self.header_lookup[columns]
        return self.data[rows, data_integer_indices]

    def __getattr__(self, item):
        if not item.startswith('_'):
            if item in self.header_lookup.keys():
                return self[:, item]
            else:
                try:
                    return self.data.__getattribute__(item)
                except AttributeError:
                    return super().__getattribute__(item)
        else:
            return super().__getattribute__(item)
