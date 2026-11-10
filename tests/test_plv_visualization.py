import numpy as np
from visualize_plv import matrix, sensors
from dataset_maker.plv_dataset import CHANNELS


def test_edge_matrix_and_sensor_order():
    edges=np.arange(325)/325
    value=matrix(edges,26)
    np.testing.assert_array_equal(value,value.T)
    np.testing.assert_array_equal(value[np.triu_indices(26,1)],edges)
    np.testing.assert_array_equal(value.diagonal(),np.ones(26))
    info,xy=sensors(CHANNELS)
    assert [n.upper() for n in info.ch_names]==CHANNELS
    assert xy.shape==(26,2) and np.isfinite(xy).all()
    # 左侧 F3 在右侧 F4 左边，额区位于枕区前方。
    assert xy[CHANNELS.index('F3'),0]<xy[CHANNELS.index('F4'),0]
    assert xy[CHANNELS.index('FZ'),1]>xy[CHANNELS.index('OZ'),1]
