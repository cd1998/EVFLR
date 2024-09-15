import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from catboost.datasets import adult
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits=load_digits()
scale = StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)
print(X_test)
print(X_test.shape)
print(y_test)
print(y_test.shape)
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
for i in range(y_train.shape[0]):
    if y_train[i] >= 5:
        y_train[i] = 1
    else:
        y_train[i] =0

for i in range(y_test.shape[0]):
    if y_test[i] >= 5:
        y_test[i] = 1
    else:
        y_test[i] =0
print(y_test)
# dftrain = pd.read_csv(r'E:\pycharm\LR\credit.csv')
# print(dftrain)
# dftrain = np.array(dftrain)
# X_train = dftrain[:,2:]
# y_train = dftrain[:,1]
#
# dftest = pd.read_csv(r'E:\pycharm\LR\credit_test.csv')
# dftest = np.array(dftest)
# X_test = dftest[:,2:]
# y_test = dftest[:,1]
#
# scale = StandardScaler()
# X_train = scale.fit_transform(X_train)
# X_test = scale.fit_transform(X_test)
# print(X_train.shape)
# print(X_train)
# print(y_train.shape)
# print(y_train)
# print(type(X_train))

# f = np.load("./Adult_processed.npz")
# X, y = f["X"], f["Y"]
# X_train = X[0:30000,:]
# y_train = y[0:30000]
# X_test = X[30001:,:]
# y_test = y[30001:]
# print(X_train)
# print(X_train.shape)
# print(y_train)
# print(y_train.shape)
# print(X_test)
# print(X_test.shape)
# print(y_test)
# print(y_test.shape)
# scale = StandardScaler()
# X_train = scale.fit_transform(X_train)
# X_test = scale.fit_transform(X_test)

# scale = StandardScaler()
# df = pd.read_csv(r'E:\pycharm\LR\nba_logreg.csv')
# df = np.array(df)
# df = df[:,1:]
# df = df.astype('float')
# print(df)
# np.random.shuffle(df)
# print(df)
# X_train = df[:,0:-1]
# y_train = df[:,-1]
# X_test = df[1001:,0:-1]
# y_test = df[1001:,-1]
#
# X_train = scale.fit_transform(X_train)
# X_test = scale.fit_transform(X_test)
#
# X_train = np.nan_to_num(X_train.astype(np.float32))
# X_test = np.nan_to_num(X_test.astype(np.float32))


model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
print('shapeeeeeeeee',model.coef_.shape)
print(222)
#print(model.predict_proba(X_test))
y_test_pred = model.predict(X_test)
print("accuracy=",accuracy_score(y_test,y_test_pred))
print("auc=",roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
print(classification_report(y_test, y_test_pred))
fpr,tpr,thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

#画曲线图
plt.figure()
plt.plot(fpr,tpr)
plt.title('$ROC curve$')
plt.show()

#计算ks
KS_max=0
best_thr=0
for i in range(len(fpr)):
    if(i==0):
        KS_max=tpr[i]-fpr[i]
        best_thr=thresholds[i]
    elif (tpr[i]-fpr[i]>KS_max):
        KS_max = tpr[i] - fpr[i]
        best_thr = thresholds[i]

print('最大KS为：',KS_max)
print('最佳阈值为：',best_thr)
