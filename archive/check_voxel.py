import os
import numpy as np
from natsort import natsorted
import pydicom

# ここの四つは各患者に応じて変更する
x_start = 160
y_start = 250
z_start = 1
slice_folder_path = "./dicom/people1/" # スライスされたdicom画像が保存されたフォルダのパス

# スライスされたdicom画像をすべて読み込む
slice_dicoms = []
for filename in natsorted(os.listdir(slice_folder_path)):
    dicom = pydicom.dcmread(os.path.join(slice_folder_path, filename))
    slice_dicoms.append(dicom)

# 32*32*32の配列を作る
voxels = np.zeros((32, 32, 32, 1), dtype=np.uint16)

# 各スライス画像を処理し、ボクセルデータに変換する
for z in range(31):
    slice_dicom = slice_dicoms[z + z_start - 1]
    slice_dicom_pixel = slice_dicom.pixel_array
    for x in range(31):
        for y in range(31):
            voxels[x + 1][y + 1][z + 1] = slice_dicom_pixel[x + x_start][y + y_start]

print(voxels.max())
# conv3dで使いやすくするために，ボクセルデータを正規化,転置する
voxels = voxels / voxels.max()
voxels = voxels.T
print(voxels.dtype)
# ボクセルデータをnpyファイルとして保存する
np.save("./people1_pl.npy", voxels)

# print(list(tes.getdata()))


# for i in range(1,31):
#     create_voxel = np.random.randint(200,300,(32,32,32,1))
#     create_voxel = create_voxel / 300
#     create_voxel = create_voxel.T
#     np.save(f"./data3/{i}_300.npy", create_voxel)

# test = np.load("./yugen.npy")
# test = torch.tensor(test)
# test = test.float()
# print(test.size())
# targets = []
# for i in range(10):
#   targets.append(1)
#   targets.append(0)
# # PyTorchのクラス内ではtorch.float型が前提
# targets = torch.tensor(targets).float()
# print(targets.dtype)

# # Trick to accept different input shapes
# x = torch.rand((1, 1) + (32,32,32))
# first_fc_in_features = 1
# print(x.size())
# for n in x.size()[1:]:
#     first_fc_in_features *= n
# x = x.view(x.size(0), -1)
# print(x.size())