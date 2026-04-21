import os
import numpy as np
from natsort import natsorted
from scipy.ndimage import convolve

path = "D:/元データ/true"

for data in natsorted(os.listdir(path)):
    reprocessing_data = np.load(f"{path}/{data}")

    # 正規化
    padding_voxel_data = np.zeros((32, 32, 32, 1), dtype=np.int32)
    voxel_data = np.load(f"{path}/{data}")
    for z in range(30):
        for x in range(30):
            for y in range(30):
                padding_voxel_data[x + 1][y + 1][z + 1] = voxel_data[x][y][z]
    reprocessing_data = (padding_voxel_data - padding_voxel_data.min()) / (padding_voxel_data.max() - padding_voxel_data.min())

    # 転置
    reprocessing_data = padding_voxel_data.T
  
    # # 平滑化フィルタを定義
    # gaussian_filter = np.array([[[[0.02058628, 0.03394104, 0.02058628], [0.03394104, 0.05595932, 0.03394104], [0.02058628, 0.03394104, 0.02058628]],
    #                             [[0.03394104, 0.05595932, 0.03394104], [0.05595932, 0.09226132, 0.05595932], [0.03394104, 0.05595932, 0.03394104]],
    #                             [[0.02058628, 0.03394104, 0.02058628], [0.03394104, 0.05595932, 0.03394104], [0.02058628, 0.03394104, 0.02058628]]]])
    # reprocessing_data = convolve(reprocessing_data, gaussian_filter, mode='constant', cval=0)

    # エッジ強調フィルタを定義
    edge_filter = np.array([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                            [[0, 1, 0], [0, -6, 0], [0, 1, 0]],
                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
    reprocessing_data = convolve(reprocessing_data, edge_filter, mode='constant', cval=0)

    np.save(f"D:/full_pro/true/{os.path.splitext(data)[0]}.npy", reprocessing_data)