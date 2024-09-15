import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
import pandas as pd
from catboost.datasets import epsilon
from torch import tensor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# scale = StandardScaler()
# cancer=load_breast_cancer()
# X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,test_size=0.2)
# #print(X_test)
# print(X_test.shape)
# print(X_test.shape[0],X_test.shape[1])
# print(y_test.shape[0],y_test[1])
# X_train = scale.fit_transform(X_train)
# X_test = scale.fit_transform(X_test)

dftrain = pd.read_csv(r'E:\pycharm\LR\credit_train.csv')
print(dftrain)
dftrain = np.array(dftrain)
X_train = dftrain[:,2:]
y_train = dftrain[:,1]

dftest = pd.read_csv(r'E:\pycharm\LR\credit_test.csv')
dftest = np.array(dftest)
X_test = dftest[:,2:]
y_test = dftest[:,1]

# print(1)
# epsilon_train, epsilon_test = epsilon()
# print(2)
#
# dftrain = np.array(epsilon_train)
# X_train = dftrain[:,1:]
# y_train = dftrain[:,0]
#
# dftest = np.array(epsilon_test)
# X_test = dftest[:,1:]
# y_test = dftest[:,0]
#
# for i in range(y_train.shape[0]):
#     if(y_train[i] == -1):
#         y_train[i] = 0
#
# for i in range(y_test.shape[0]):
#     if(y_test[i] == -1):
#         y_test[i] = 0

X_train = torch.as_tensor(X_train,dtype=torch.float32)
y_train = torch.as_tensor(y_train,dtype=torch.float32)
X_test = torch.as_tensor(X_test,dtype=torch.float32)
y_test = torch.as_tensor(y_test,dtype=torch.float32)
print(y_train.shape)
print(y_train)
print(X_train.shape)
print(X_train)

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(X_train.shape[1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr_net = LR()   # 实例化逻辑回归模型

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(lr_net.parameters(),lr=0.15)
for iteration in range(100):
    # 前向传播
    y_pred = lr_net(X_train)

    # 计算 loss
    loss = loss_fn(y_pred.squeeze(), y_train)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 更新参数
    optimizer.step()

    y_pred_test = lr_net(X_test)
    mask = y_pred_test.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
    y_pred_test_1dim = []
    for i in range(y_test.shape[0]):
        y_pred_test_1dim.append(y_pred_test[i].item())
    correct = (mask == y_test).sum()  # 计算正确预测的样本个数
    acc = correct.item() / y_test.size(0)  # 计算分类准确率
    print(iteration, acc)
    print(iteration, "AUC", roc_auc_score(y_test, y_pred_test_1dim))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_1dim)
    # 计算ks
    KS_max = 0
    best_thr = 0
    for i in range(len(fpr)):
        if (i == 0):
            KS_max = tpr[i] - fpr[i]
            best_thr = thresholds[i]
        elif (tpr[i] - fpr[i] > KS_max):
            KS_max = tpr[i] - fpr[i]
            best_thr = thresholds[i]

    print('KS为：', KS_max)
