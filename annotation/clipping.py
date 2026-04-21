import os
import numpy as np
from natsort import natsorted
import pydicom

def clipping(x_center, y_center, z_center, slice_folder_path):
    # スライスされたdicom画像をすべて読み込む
    slice_dicoms = []
    for filename in natsorted(os.listdir(slice_folder_path)):
        dicom = pydicom.dcmread(os.path.join(slice_folder_path, filename))
        slice_dicoms.append(dicom)

    voxels = np.zeros((32, 32, 32, 1), dtype=np.uint16)

    # 各スライス画像を処理し、ボクセルデータに変換する
    for z in range(0, 32):
        slice_dicom = slice_dicoms[z + z_center - 16]
        slice_dicom_pixel = slice_dicom.pixel_array
        for x in range(0, 32):
            for y in range(0, 32):
                voxels[x][y][z] = slice_dicom_pixel[x + x_center - 16][y + y_center - 16]

    # conv3dで使いやすくするために，ボクセルデータを正規化,転置する
    voxels = (voxels - voxels.min()) / (voxels.max() - voxels.min())
    voxels = voxels.T
    np.save(f"./{os.path.basename(slice_folder_path)}_pl.npy", voxels)