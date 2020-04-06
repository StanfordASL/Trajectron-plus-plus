import numpy as np
import pandas as pd
from data import SingleHeaderNumpyArray, DoubleHeaderNumpyArray


def test_single_header_numpy_array():
    x = np.random.rand(10)
    y = np.random.rand(10)

    array = SingleHeaderNumpyArray(np.stack((x, y), axis=-1), ['x', 'y'])

    assert (array[:, 'x'] == x).all()
    assert (array[:, 'y'] == y).all()
    assert (array[3:7, 'y'] == y[3:7]).all()
    assert (array.x == x).all()
    assert (array.y == y).all()


def test_double_header_numpy_array():
    x = np.random.rand(10)
    y = np.random.rand(10)
    vx = np.random.rand(10)
    vy = np.random.rand(10)

    data_dict = {('position', 'x'): x,
                 ('position', 'y'): y,
                 ('velocity', 'x'): vx,
                 ('velocity', 'y'): vy}

    data_columns = pd.MultiIndex.from_product([['position', 'velocity'], ['x', 'y']])

    node_data = pd.DataFrame(data_dict, columns=data_columns)

    array = DoubleHeaderNumpyArray(node_data.values, list(node_data.columns))

    test_header_dict = {'position': ['x', 'y'], 'velocity': ['y']}

    assert (array[:, ('position', 'x')] == x).all()
    assert (array[:, ('velocity', 'y')] == vy).all()
    assert (array[4:7, ('velocity', 'y')] == vy[4:7]).all()
    assert (array[:, [('position', 'x'), ('velocity', 'y')]] == np.stack((x, vy), axis=-1)).all()
    assert (array[:, [('position', 'y'), ('velocity', 'x')]] == np.stack((y, vx), axis=-1)).all()
    assert (array[2:6, [('position', 'y'), ('velocity', 'x')]] == np.stack((y, vx), axis=-1)[2:6]).all()
    assert (array[:, test_header_dict] == np.stack((x, y, vy), axis=-1)).all()
    assert (array[1:8, test_header_dict] == np.stack((x, y, vy), axis=-1)[1:8]).all()
    assert (array.position.x == x).all()


