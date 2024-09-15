import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

dftrain = pd.read_csv(r'E:\pycharm\LR\give_credit_train.csv')
print(dftrain)
dftrain = np.array(dftrain)
X_train = dftrain[:, 2:]
y_train = dftrain[:, 1]

dftest = pd.read_csv(r'E:\pycharm\LR\give_credit_test.csv')
dftest = np.array(dftest)
X_test = dftest[:, 2:]
y_test = dftest[:, 1]

print(X_train.shape)

config = {
        'n_iter': 10000,
        'lambda': 0,
        'eta': 0.5,
    }

theta = 2*np.random.random(X_train.shape[1])-1
print(theta)

normal_loss_list = []

## 开始训练, 根据配置的迭代次数
for i in range(config['n_iter']):

    # 计算梯度
    dl = 0
    for j in range(X_train.shape[0]):
        tmp = 1 / (1 + np.exp(y_train[j] * X_train[j, :].dot(theta)))
        dl += -tmp * y_train[j] * X_train[j, :]
    dl = dl + config['lambda'] * theta
    # 计算损失(去掉惩罚项)
    normal_loss = np.sum(np.log(1 + np.exp(-y_train * X_train.dot(theta)))) / X_train.shape[0]
    normal_loss_list.append(normal_loss)

    # 更新theta
    theta = theta - config['eta'] * dl / X_train.shape[0]

    normal_y_pre_prob = 1 / (1 + np.exp(-X_test.dot(theta)))
    print(normal_y_pre_prob)
    normal_y_pre = np.where(normal_y_pre_prob > 0.5, 1, 0)
    print(normal_y_pre)
    print("test normal lr acc", accuracy_score(y_test, normal_y_pre))
    print("test normal lr auc", roc_auc_score(y_test, normal_y_pre_prob))

