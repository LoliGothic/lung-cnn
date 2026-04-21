import os
import numpy as np
from natsort import natsorted

path = "./annotation_data/true"

for data in natsorted(os.listdir(path)):
    padding_voxel_data = np.zeros((32, 32, 32, 1), dtype=np.int32)
    voxel_data = np.load(f"{path}/{data}")

    for z in range(30):
        for x in range(30):
            for y in range(30):
                padding_voxel_data[x + 1][y + 1][z + 1] = voxel_data[x][y][z]

    padding_voxel_data = (padding_voxel_data - padding_voxel_data.min()) / (padding_voxel_data.max() - padding_voxel_data.min())
    padding_voxel_data = padding_voxel_data.T

    np.save(f"./reprocessing_data/true/{os.path.splitext(data)[0]}.npy", padding_voxel_data)

    print(data)
    print(padding_voxel_data.max())
    print(padding_voxel_data.shape)