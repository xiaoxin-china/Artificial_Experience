import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
#导入分层抽样工具
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

#-----------------------------
#参数
batch_size = 16
hidden_size = 128
LR = 0.001
epoches = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-----------------------------


#-----------------------------
#数据预处理（保留month列，用于分层抽样）
def data_processing(path = 'temps.csv'):
    features = pd.read_csv(path)
    # 保存原始month列（用于分层抽样，不参与模型训练，只是划分依据）
    month_feature = features['month'].values  # 提取月份特征
    # 将非数值特征例如这里的周几独热编码
    features = pd.get_dummies(features)
    #标签（y）
    labels = np.array(features['actual'])
    #在特征中去除标签（x）
    features = features.drop('actual',axis = 1)
    #列的名字单独保存一下
    features_names = list(features.columns)
    features = np.array(features)
    input_features = preprocessing.StandardScaler().fit_transform(features)#各个特征之间标准化

    x = torch.tensor(input_features,dtype=torch.float)  #先放CPU，抽样后再传GPU
    y = torch.tensor(labels,dtype=torch.float).reshape(-1,1)
    return x,y,features_names, month_feature  #返回month_feature用于分层

# 加载数据+获取月份特征
x,y,features_names, month_feature = data_processing('temps.csv')
input_size = x.shape[1]
total_samples = x.shape[0]
print(f"输入特征维度：{input_size},样本总数：{total_samples},特征列名：{features_names}")

#按月份分层抽样划分训练集/测试集（7:3划分，保证季节分布一致）
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, test_idx in split.split(x.cpu().numpy(), month_feature):  # 分层依据是month_feature
    # 按抽样索引切分张量
    x_train = x[train_idx].to(device)
    y_train = y[train_idx].to(device)
    x_test = x[test_idx].to(device)
    y_test = y[test_idx].to(device)

# 验证分层效果：打印训练集/测试集的月份分布
print("\n训练集月份分布：")
print(pd.Series(month_feature[train_idx]).value_counts(normalize=True).sort_index())
print("\n测试集月份分布：")
print(pd.Series(month_feature[test_idx]).value_counts(normalize=True).sort_index())
print(f"\n训练集样本数：{x_train.shape[0]}, 测试集样本数：{x_test.shape[0]}")
#-----------------------------


#-----------------------------
#数据集封装（完全不变）
def data_dataloader(x,y,batch_size=16):
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = True)
    return dataloader

# 封装训练/测试集加载器
train_dl = data_dataloader(x_train, y_train, batch_size)
test_dl = data_dataloader(x_test, y_test, batch_size)
#-----------------------------


#-----------------------------
#模型定义（完全不变，修复激活函数命名混淆）
class TempRegressionModel(nn.Module):
    def __init__(self,input_size,hidden_size1=128,hidden_size2=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

#模型实例化（修正拼写：bulid→build）
def build_model(input_size,hidden_size1=128,hidden_size2=256):
    model = TempRegressionModel(input_size,hidden_size1,hidden_size2).to(device)
    return model

model = build_model(input_size,hidden_size)
#-----------------------------



#-----------------------------
#训练配置（完全不变）
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(),lr = LR)
# ✅ 可选优化：加学习率调度器，解决后期收敛震荡
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
#-----------------------------



#-----------------------------
#模型训练（加学习率调度器）
def train_model(model,dataloader,criterion,optimizer,epochs, scheduler):
    model.train()
    losses = []
    for epoch in range(epochs):
        batch_loss = []
        for xb,yb in dataloader:
            pred = model(xb)
            loss = criterion(pred,yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        scheduler.step()  # 更新学习率
        if (epoch+1) % 100 == 0:
            avg_loss = np.mean(batch_loss)
            losses.append(avg_loss)
            print(f"Epoch [{epoch + 1:4d}/{epochs}] | 训练集平均MSE损失：{avg_loss:.4f}")
    return losses
# 执行训练
train_losses = train_model(model, train_dl, criterion, optimizer, epoches, scheduler)
#-----------------------------


# ===================== 模型预测与结果分析（完全不变，基于测试集） =====================
def model_predict(model, x, y, criterion, device):
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x)
        total_mse = criterion(y_pred_tensor, y).item()
        total_rmse = np.sqrt(total_mse)
        y_true = y.cpu().numpy().ravel()
        y_pred = y_pred_tensor.cpu().numpy().ravel()

    print("=" * 50)
    print(f"测试集预测结果整体评估（季节分布均衡）")
    print(f"测试集MSE损失：{total_mse:.4f}")
    print(f"测试集RMSE（均方根误差）：{total_rmse:.4f} ℃")
    print("=" * 50)
    return y_true, y_pred, total_mse, total_rmse

# 测试集预测
y_true, y_pred, pred_mse, pred_rmse = model_predict(model, x_test, y_test, criterion, device)

# ===================== 预测结果可视化（完全不变） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图1：测试集前100个样本 真实值vs预测值
plt.figure(figsize=(12, 4), dpi=100)
plt.plot(y_true[:100], label='真实气温', color='royalblue', linewidth=2)
plt.plot(y_pred[:100], label='预测气温', color='crimson', linewidth=2, linestyle='--')
plt.xlabel('测试集样本序号', fontsize=12)
plt.ylabel('气温 (℃)', fontsize=12)
plt.title(f'测试集前100个样本 - 真实气温vs预测气温（RMSE：{pred_rmse:.2f}℃）', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 图2：测试集全量样本 散点图+拟合线
plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(y_true, y_pred, color='lightseagreen', alpha=0.6, s=20)
min_val, max_val = min(y_true), max(y_true)
plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='完美拟合线')
plt.xlabel('真实气温 (℃)', fontsize=12)
plt.ylabel('预测气温 (℃)', fontsize=12)
plt.title(f'测试集全量样本 - 真实气温vs预测气温（RMSE：{pred_rmse:.2f}℃）', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()