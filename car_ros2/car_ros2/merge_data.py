import numpy as np
import pickle

datas = []
max_len = 241
target_features = 39

# List of files to merge
file_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in file_indices:
    file_path = f'data/data{i}_multi_mpc.pkl'
    with open(file_path, 'rb') as f:
        d = pickle.load(f)

    # Handle data if stored as an array of objects
    if isinstance(d, np.ndarray) and d.dtype == object:
        d = [np.array(x) for x in d]

    fixed = []
    for idx, x in enumerate(d):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Pad or truncate time dimension
        if x.shape[0] >= max_len:
            x = x[:max_len, :]
        else:
            pad = np.zeros((max_len - x.shape[0], x.shape[1]))
            x = np.vstack([x, pad])

        # Pad or truncate feature dimension
        if x.shape[1] > target_features:
            x = x[:, :target_features]
        elif x.shape[1] < target_features:
            pad = np.zeros((x.shape[0], target_features - x.shape[1]))
            x = np.hstack([x, pad])

        print(f"File {i}, Episode {idx} final shape: {x.shape}")
        fixed.append(x)

    d = np.stack(fixed, axis=0)
    print(f"Loaded file {i}: shape {d.shape}")

    datas.append(d)

# Concatenate along episode dimension
data_large = np.concatenate(datas, axis=0)
print("Final concatenated shape:", data_large.shape)

# Save merged data to pickle file
output_path = 'data/data_large_multi_mpc.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(data_large, f)

print(f"Saved merged data to {output_path}")