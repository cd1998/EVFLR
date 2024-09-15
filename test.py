import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn.datasets import load_digits
from catboost.datasets import epsilon
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

config = {
        'n_iter': 60,
        'lambda': 0.1, #0.1
        'lr': 0.05,     #0.05
        'factor12': 2047,
        'alpha12': 4.89,
        'factor16': 32768,
        'alpha16': 5.83,
        'factor20': 524288,
        'alpha20': 6.66
    }


# digits=load_digits()
# scale = StandardScaler()
# X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target)
#
# print(X_train)
# print(X_train.shape)
# print(X_test)
# print(X_test.shape)
# print(y_test)
# print(y_test.shape)
# X_train = scale.fit_transform(X_train)
# X_test = scale.fit_transform(X_test)
# for i in range(y_train.shape[0]):
#     if y_train[i] >= 5:
#         y_train[i] = 1
#     else:
#         y_train[i] =-1
#
# for i in range(y_test.shape[0]):
#     if y_test[i] >= 5:
#         y_test[i] = 1
#     else:
#         y_test[i] =0
# print(X_train)
# print(y_train)


# scale = StandardScaler()
# cancer=load_breast_cancer()
# X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,test_size=0.2)
# #print(X_test)
# print(X_test.shape)
# print(X_test.shape[0],X_test.shape[1])
# print(y_test.shape[0],y_test[1])
# X_train = scale.fit_transform(X_train)
# X_test = scale.fit_transform(X_test)


f = np.load("./Adult_processed.npz")
X, y = f["X"], f["Y"]
X_train = X[0:30000,:]
y_train = y[0:30000]
X_test = X[30001:,:]
y_test = y[30001:]
for i in range(y_train.shape[0]):
    if y_train[i] == 0:
        y_train[i] = -1

print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)
print(X_test)
print(X_test.shape)
print(y_test)
print(y_test.shape)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)


X_train_clipping_12 = np.where((X_train>-config['alpha12']) & (X_train<config['alpha12']), X_train, np.sign(X_train)*config['alpha12'])
X_train_clipping_16 = np.where((X_train>-config['alpha16']) & (X_train<config['alpha16']), X_train, np.sign(X_train)*config['alpha16'])
X_train_clipping_20 = np.where((X_train>-config['alpha20']) & (X_train<config['alpha20']), X_train, np.sign(X_train)*config['alpha20'])
#print(X_train_clipping)
X_train_integer_12 = np.around(X_train_clipping_12 * config['factor12'] / config['alpha12'])
X_train_integer_16 = np.around(X_train_clipping_16 * config['factor16'] / config['alpha16'])
X_train_integer_20 = np.around(X_train_clipping_20 * config['factor20'] / config['alpha20'])
#print(X_train_integer)

w = 2*np.random.random(X_train.shape[1])-1 #一组[-1,1]之间的随机浮点数
#w = np.zeros(X_train.shape[1])              #一组全0数
#w = np.random.randn(X_train.shape[1])      #一组服从正态分布的浮点数
w_ori = w
w_12 = w
w_16 = w
w_20 = w
AUC = []
AUC_ori = []
AUC_12 = []
AUC_16 = []
AUC_20 = []
print("w:",w)
for j in range(config['n_iter']):
    wx = np.matmul(X_train, w)
    wx_12 = np.matmul(X_train_clipping_12, w_12)
    wx_16 = np.matmul(X_train_clipping_16, w_16)
    wx_20 = np.matmul(X_train_clipping_20, w_20)
    print('wx:',wx)
    print('wx_12:', wx_12)
    print('wx_16:', wx_16)
    print('wx_20:', wx_20)

    d = 0.25 * (wx - 2 * y_train)
    d12 = 0.25 * (wx_12 - 2 * y_train)
    d16 = 0.25 * (wx_16 - 2 * y_train)
    d20 = 0.25 * (wx_20 - 2 * y_train)
    print('d:',d)
    print('d_shape',np.shape(d))
    # d_ori = 1 / (1 + np.exp(-X_train.dot(w)))
    # print('d_ori',d_ori)
    # print('d_ori_shape', np.shape(d_ori))
    d_12 = np.around(d12 * config['factor12'] / (config['alpha12'] * X_train.shape[1]))
    d_16 = np.around(d16 * config['factor16'] / (config['alpha16'] * X_train.shape[1]))
    d_20 = np.around(d20 * config['factor20'] / (config['alpha20'] * X_train.shape[1]))
    #print("d:", d)
    gradient = np.zeros(X_train.shape[1])
    gradient_12 = np.zeros(X_train.shape[1])
    gradient_16 = np.zeros(X_train.shape[1])
    gradient_20 = np.zeros(X_train.shape[1])
    for i in range(X_train.shape[0]):
        gradient += 1 / X_train.shape[0] * (d[i] * X_train[i])
        gradient_12 += 1 / X_train.shape[0] * (d_12[i] * X_train_integer_12[i])
        gradient_16 += 1 / X_train.shape[0] * (d_16[i] * X_train_integer_16[i])
        gradient_20 += 1 / X_train.shape[0] * (d_20[i] * X_train_integer_20[i])
    print("gradient:", gradient)
    print('gradientt:',gradient * X_train.shape[0])
    # print("gradient_12:", gradient_12)
    # print("gradient_16:", gradient_16)

    gradient_ori = np.zeros(X_train.shape[1])
    wx_ori = np.matmul(X_train, w_ori)
    print('wx_ori',wx_ori)
    for i in range(X_train.shape[0]):
        tmp = y_train[i] * wx_ori[i]
        # print('tmp:',tmp)
        # print(1 / (1 + np.exp(-tmp)) * (-y_train[i] * X_train[i]))
        #gradient_ori += 1 / X_train.shape[0] * (1 / (1 + np.exp(tmp)) * (-y_train[i] * X_train[i]))  #y=-1/1
        gradient_ori += 1 / X_train.shape[0] * (( 1 / (1+np.exp(-wx_ori[i])) - y_train[i] ) * X_train[i]) #y=0/1
    print('gradient_ori:',gradient_ori)
    # g = gradient
    # g8 = gradient_8 * (np.square(config['alpha8']) * X_train.shape[1]) / np.square(config['factor8'])
    # g16 = gradient_16 * (np.square(config['alpha16']) * X_train.shape[1]) / np.square(config['factor16'])
    # g20 = gradient_20 * (np.square(config['alpha20']) * X_train.shape[1]) / config['factor20']**2
    loss = 0
    for i in range(X_train.shape[0]):
        loss += np.log(2) - 0.5 * y_train[i] * np.matmul(X_train[i], w) + 0.125 * np.square(np.matmul(X_train[i], w))
    loss = loss / X_train.shape[0]
    print("loss:", loss)
    w = w - config['lr'] * gradient - config['lambda']*w
    w_12 = w_12 - config['lr'] * gradient_12 * (np.square(config['alpha12']) * X_train.shape[1]) / np.square(config['factor12']) - config['lambda'] * w_12
    w_16 = w_16 - config['lr'] * gradient_16 * (np.square(config['alpha16']) * X_train.shape[1]) / np.square(config['factor16']) - config['lambda'] * w_16
    w_20 = w_20 - config['lr'] * gradient_20 * (np.square(config['alpha20']) * X_train.shape[1]) / config['factor20']**2 - config['lambda'] * w_20
    w_ori = w_ori - config['lr'] * gradient_ori - config['lambda']*w_ori
    print("w",w)
    # print("w12",w_12)
    # print("w16",w_16)
    print("w_ori:",w_ori)

    y_test_pre_prob = 1 / (1 + np.exp(-X_test.dot(w)))
    y_test_pre_prob_12 = 1 / (1 + np.exp(-X_test.dot(w_12)))
    y_test_pre_prob_16 = 1 / (1 + np.exp(-X_test.dot(w_16)))
    y_test_pre_prob_20 = 1 / (1 + np.exp(-X_test.dot(w_20)))
    y_test_pre_prob_ori = 1 / (1 + np.exp(-X_test.dot(w_ori)))

    y_test_pre = np.floor(y_test_pre_prob + 0.5)
    y_test_pre_12 = np.floor(y_test_pre_prob_12 + 0.5)
    y_test_pre_16 = np.floor(y_test_pre_prob_16 + 0.5)
    y_test_pre_20 = np.floor(y_test_pre_prob_20 + 0.5)
    y_test_pre_ori = np.floor(y_test_pre_prob_ori + 0.5)

    print("acc:",accuracy_score(y_test,y_test_pre))
    print("acc_12:", accuracy_score(y_test, y_test_pre_12))
    print("acc_16:", accuracy_score(y_test, y_test_pre_16))
    print("acc_20:", accuracy_score(y_test, y_test_pre_20))
    print("acc_ori:", accuracy_score(y_test, y_test_pre_ori))

    print("AUC:",roc_auc_score(y_test,y_test_pre_prob))
    print("AUC_12:", roc_auc_score(y_test, y_test_pre_prob_12))
    print("AUC_16:", roc_auc_score(y_test, y_test_pre_prob_16))
    print("AUC_20:", roc_auc_score(y_test, y_test_pre_prob_20))
    print("AUC_ori:", roc_auc_score(y_test, y_test_pre_prob_ori))

    fpr, tpr, thresholds = roc_curve(y_test, y_test_pre_prob)
    fpr12, tpr12, thresholds12 = roc_curve(y_test, y_test_pre_prob_12)
    fpr16, tpr16, thresholds16 = roc_curve(y_test, y_test_pre_prob_16)
    fpr20, tpr20, thresholds20 = roc_curve(y_test, y_test_pre_prob_20)
    fprori, tprori, thresholdsori = roc_curve(y_test, y_test_pre_prob_ori)

    KS = 0
    best_thr = 0
    for i in range(len(fpr)):
        if (i == 0):
            KS = tpr[i] - fpr[i]
            best_thr = thresholds[i]
        elif (tpr[i] - fpr[i] > KS):
            KS = tpr[i] - fpr[i]
            best_thr = thresholds[i]
    print('KS：', KS)

    KS12 = 0
    best_thr12 = 0
    for i in range(len(fpr12)):
        if (i == 0):
            KS12 = tpr12[i] - fpr12[i]
            best_thr12 = thresholds12[i]
        elif (tpr12[i] - fpr12[i] > KS12):
            KS12 = tpr12[i] - fpr12[i]
            best_thr12 = thresholds12[i]
    print('KS12：', KS12)

    KS16 = 0
    best_thr16 = 0
    for i in range(len(fpr16)):
        if (i == 0):
            KS16 = tpr16[i] - fpr16[i]
            best_thr16 = thresholds16[i]
        elif (tpr16[i] - fpr16[i] > KS16):
            KS16 = tpr16[i] - fpr16[i]
            best_thr16 = thresholds16[i]
    print('KS16：', KS16)

    KS20 = 0
    best_thr20 = 0
    for i in range(len(fpr20)):
        if (i == 0):
            KS20 = tpr20[i] - fpr20[i]
            best_thr20 = thresholds20[i]
        elif (tpr20[i] - fpr20[i] > KS20):
            KS20 = tpr20[i] - fpr20[i]
            best_thr20 = thresholds20[i]
    print('KS20：', KS20)

    KSori = 0
    best_throri = 0
    for i in range(len(fprori)):
        if (i == 0):
            KSori = tprori[i] - fprori[i]
            best_throri = thresholdsori[i]
        elif (tprori[i] - fprori[i] > KSori):
            KSori = tprori[i] - fprori[i]
            best_throri = thresholdsori[i]
    print('KSori：', KSori)
    print("\n")

    AUC.append(roc_auc_score(y_test,y_test_pre_prob))
    AUC_12.append(roc_auc_score(y_test,y_test_pre_prob_12))
    AUC_16.append(roc_auc_score(y_test, y_test_pre_prob_16))
    AUC_20.append(roc_auc_score(y_test, y_test_pre_prob_20))
    AUC_ori.append(roc_auc_score(y_test, y_test_pre_prob_ori))
#
#
# # print(AUC)
#
#plt.figure(figsize=(6,9))  # 图的大小
# plt.xlabel('epoch',fontsize=18)
# plt.ylabel('AUC',fontsize=18)
plt.yticks(size=20)
plt.xticks(size=20)
#画曲线图
x = np.arange(0,config['n_iter'],1)
#plt.figure()
plt.plot(x,AUC,color='yellowgreen',label='plain',linewidth='4')
plt.plot(x,AUC_12,color='orange',label='12 bits',linewidth='3')
plt.plot(x,AUC_16,color='cyan',label='16 bits',linewidth='2')
plt.plot(x,AUC_20,color='cornflowerblue',label='20 bits',linewidth='1')
#plt.plot(x,AUC_ori,color='red',label='ori',linewidth='5')
plt.legend(loc='lower right', fontsize=20)
plt.savefig('4.eps')
plt.show()