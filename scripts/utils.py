import numpy as np

def get_xy_from_npz(file_path, x_key='X', y_key='y'):
    npz_data = np.load(file_path, allow_pickle=True)
    available_keys = npz_data.files
    if x_key not in available_keys:
        raise KeyError(f"X key '{x_key}' not found in NPZ file. Available keys: {available_keys}")
    if y_key not in available_keys:
        raise KeyError(f"y key '{y_key}' not found in NPZ file. Available keys: {available_keys}")
    X = npz_data[x_key]
    y = npz_data[y_key]
    return X, y