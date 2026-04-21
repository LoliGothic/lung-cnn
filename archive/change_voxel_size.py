import numpy as np
from scipy.interpolate import RegularGridInterpolator

voxel_data = np.load("./343#.npy")
print(voxel_data[211][247][15])

# ボクセルデータの座標を作成する
x = np.arange(0, 512)
y = np.arange(0, 512)
z = np.arange(0, voxel_data.shape[2])

# RegularGridInterpolatorを作成する(元となる三次元データを作成)
interpolator = RegularGridInterpolator((x, y, z), voxel_data, method="cubic")

# 30×30×30の新しいボクセルデータの座標を作成する
new_x = np.linspace(0, 511, 30)
new_y = np.linspace(0, 511, 30)
new_z = np.linspace(0, voxel_data.shape[2] - 1, 30)

# 新しいボクセルデータの座標をメッシュグリッドの座標に変換する
xg, yg, zg = np.meshgrid(new_x, new_y, new_z, indexing="ij")

# 0パディングされた32*32*32のボクセルデータを作る
voxel_data_32 = np.zeros((32, 32, 32, 1), dtype=np.uint16)

# 0パディングされた32*32*32のボクセルデータを作る
voxel_data_32 = np.zeros((32, 32, 32, 1), dtype=np.uint8)

# RegularGridInterpolatorを使用して新しいボクセルデータを補間する
for z in range(8,9):
    for x in range(12,13):
        for y in range(14,15):
            voxel_data_32[x + 1][y + 1][z + 1] = interpolator((xg[x][y][z], yg[x][y][z], zg[x][y][z]))
            print(xg[x][y][z], yg[x][y][z], zg[x][y][z])
            print(interpolator((xg[x][y][z], yg[x][y][z], zg[x][y][z]))[0])

# リサイズされたボクセルデータのサイズを確認する
print(voxel_data_32.shape)