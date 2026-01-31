from pathlib import Path
import requests
import pickle
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

# 1. 下载加载MNIST
DATA_PATH = Path("data") / "mnist"
DATA_PATH.mkdir(parents=True, exist_ok=True)
URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (DATA_PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (DATA_PATH / FILENAME).open("wb").write(content)

with gzip.open((DATA_PATH / FILENAME).as_posix(), "rb") as f:
    (x_train, y_train), (x_valid, y_valid), _ = pickle.load(f, encoding="latin-1")

# 2. 转张量
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

# 3. 定义模型
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)

# 4. 数据加载器
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2)
    )

# 5. 训练工具函数
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)

def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(f"Step {step+1}: Val Loss {val_loss:.4f}")

# 6. 初始化+训练
bs = 64
train_dl, valid_dl = get_data(TensorDataset(x_train, y_train), TensorDataset(x_valid, y_valid), bs)
model, opt = get_model()
fit(25, model, F.cross_entropy, opt, train_dl, valid_dl)