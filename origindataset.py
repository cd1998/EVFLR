import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from catboost.datasets import epsilon
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

print(1)
epsilon_train, epsilon_test = epsilon()
print(2)

dftrain = np.array(epsilon_train)
X_train = dftrain[:,1:]
y_train = dftrain[:,0]

dftest = np.array(epsilon_test)
X_test = dftest[:,1:]
y_test = dftest[:,0]

print(y_test)
print(X_test)
print(X_test.shape)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_test_predicted = model.predict(X_test)
print("accuracy=",accuracy_score(y_test,y_test_predicted))
print("auc=",roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
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

print('KS为：',KS_max)
print('最佳阈值为：',best_thr)