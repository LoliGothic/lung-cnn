import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.voxnet import VoxNet
from models.voxnet2 import VoxNet2
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from natsort import natsorted # 普通のsortで文字列900,1000をsortすると，1000,900となってしまうため，natsortを使う

# データセットを定義する
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

# トレーニング関数を定義する
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 予測値と正解ラベルから、真陽性、真陰性、偽陽性、偽陰性を計算する
        true_positives += ((predicted == 1) & (labels == 1)).sum().item()
        true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
        false_positives += ((predicted == 1) & (labels == 0)).sum().item()
        false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = correct / total
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0

    try:
        specificity = true_negatives / (true_negatives + false_positives)
    except ZeroDivisionError:
        specificity = 0
    
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0

    return running_loss / len(dataloader), accuracy, precision, recall, specificity, f1_score

# テスト関数を定義する
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 予測値と正解ラベルから、真陽性、真陰性、偽陽性、偽陰性を計算する
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = correct / total
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0

    try:
        specificity = true_negatives / (true_negatives + false_positives)
    except ZeroDivisionError:
        specificity = 0
    
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0

    return running_loss / len(dataloader), accuracy, precision, recall, specificity, f1_score

# パラメータを設定する
learning_rate = 0.0001
batch_size = 50 #適当に変えた所
num_epochs = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# データセットを作成する
data = []  # 入力データのdepthは合わせる必要がある?
targets = []

true_data_path = f"./production_data/true"
false_data_path = f"./production_data/false"

# データをdataとtargetsに入れる
for true_data in natsorted(os.listdir(true_data_path)):
    infiltration = np.load(f"{true_data_path}/{true_data}")
    data.append(infiltration)
    targets.append(1)
for false_data in natsorted(os.listdir(false_data_path)):
    not_infiltration = np.load(f"{false_data_path}/{false_data}")
    data.append(not_infiltration)
    targets.append(0)

# npの配列に変換
data = np.array(data)
targets = np.array(targets)

# PyTorchのtorch.transform.ToTensor(data)クラス内ではtorch.float型が前提
data = torch.tensor(data).float()
    
# 正解ラベルはtorch.long型
targets = torch.tensor(targets).long()

dataset = MyDataset(data, targets)

# トレーニングセットとテストセットに分割する
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-10, 10])
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# グラフを書くためのx座標とy座標
train_loss_graph_x = []
train_loss_graph_y = []
test_accuracy_graph_x = []
test_accuracy_graph_y = []

# 交差検証の設定(五分割，シャッフル)
kf = KFold(n_splits=5, shuffle=True)

for _fold, (train_index, test_index) in enumerate(kf.split(range(len(dataset)))):
    # モデル、オプティマイザ、損失関数を定義する(リセットする)
    # model = VoxNet().to(device)
    model = VoxNet2().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    # トレーニングを実行する
    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_precision, train_recall, train_specificity, train_f1_score = train(model, train_dataloader, optimizer, criterion, device)
        test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1_score = test(model, test_dataloader, criterion, device)

        train_loss_graph_x.append(epoch)
        train_loss_graph_y.append(train_loss)
        test_accuracy_graph_x.append(epoch)
        test_accuracy_graph_y.append(test_accuracy)

        # トレーニングのloss,テストのloss,acc,pre,rec,spe,f1を表示
        # print(f"Epoch {epoch+1}: Train Acc = {train_accuracy}")
        # print(f"Epoch {epoch+1}: Test Acc = {test_accuracy}")
        # print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Train Precision = {train_precision:.4f}, Train Recall = {train_recall:.4f}, Train Specificity = {train_specificity:.4f}, Train F1 Score = {train_f1_score:.4f}")
        # if epoch == 49 or epoch == 99 or epoch == 149 or epoch == 199 or epoch == 249 or epoch == 299:
        print(f"Epoch {epoch+1}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}, Test Precision = {test_precision:.4f}, Test Recall = {test_recall:.4f}, Test Specificity = {test_specificity:.4f}, Test F1 Score = {test_f1_score:.4f}")

# グラフを書いてカレントディレクトリに保存
plt.plot(train_loss_graph_x, train_loss_graph_y)
plt.savefig("./result/train_loss.png")

plt.plot(test_accuracy_graph_x, test_accuracy_graph_y)
plt.savefig("./result/test_accuracy.png")

# モデルを保存する
torch.save(model.state_dict(), "voxnet_model.pt")
