from pathlib import Path
from typing import Union

import numpy as np
import h5py

def h5_to_np_array(inpath: Path) -> Union[np.ndarray, None]:
    assert inpath.suffix == '.h5'
    assert inpath.exists()

    try:
        with h5py.File(str(inpath), 'r') as h5f:
            array = np.asarray(h5f['voxel_grid'])
            return array
    except OSError as e:
        print(f'Error loading {inpath}')
    return None

def np_array_to_h5(array: np.ndarray, outpath: Path) -> None:
    isinstance(array, np.ndarray)
    assert outpath.suffix == '.h5'

    with h5py.File(str(outpath), 'w') as h5f:
        h5f.create_dataset('voxel_grid', data=array, shape=array.shape, dtype=array.dtype,
                           **_blosc_opts(complevel=1, shuffle='byte'))
        
def _blosc_opts(complevel=1, complib='blosc:zstd', shuffle='byte'):
    shuffle = 2 if shuffle == 'bit' else 1 if shuffle == 'byte' else 0
    compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
    complib = ['blosc:' + c for c in compressors].index(complib)
    args = {
        'compression': 32001,
        'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib),
    }
    if shuffle > 0:
        # Do not use h5py shuffle if blosc shuffle is enabled.
        args['shuffle'] = False
    return args

def check_key_and_bool(config: dict, key: str) -> bool:
    return key in config.keys() and config[key]