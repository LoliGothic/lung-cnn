import os
import pydicom
import numpy as np
from natsort import natsorted # 普通のsortで文字列900,1000をsortすると，1000,900となってしまうため，natsortを使う
import matplotlib.pyplot as plt

# 300番台~1700番台までのスライスされたdicom画像が保存されているディレクトリのパス
all_patient_path = "./data"

x = [0] * 1000
y = [0] * 1000

tmp = []

# x座標を0~1000にする
for i in range(1000):
    x[i] = i

# 300番台の人,400番台の人...1700番台の人とループを回す
for divide_patient_path in natsorted(os.listdir(all_patient_path)):
    # 300番,301番...1798番,1799番とループを回す
    for patient_path in natsorted(os.listdir(path=f"{all_patient_path}/{divide_patient_path}")):

        if len(natsorted(os.listdir(path=f"{all_patient_path}/{divide_patient_path}/{patient_path}"))) == 979:
            print(patient_path)
        tmp.append(len(natsorted(os.listdir(path=f"{all_patient_path}/{divide_patient_path}/{patient_path}"))))
        # 1患者あたり何枚のdicom画像があるかをy座標にする
        y[len(natsorted(os.listdir(path=f"{all_patient_path}/{divide_patient_path}/{patient_path}")))] += 1

print(max(tmp))
# 1患者あたり何枚のdicom画像があるかグラフを生成し保存する
plt.bar(x,y)
plt.savefig("distribution.png")