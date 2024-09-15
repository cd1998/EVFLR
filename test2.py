from catboost.datasets import amazon
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

dftrain = pd.read_csv(r'E:\pycharm\LR\credit-data-trainingset.csv')
print(dftrain)
dftrain = np.array(dftrain)
X_train = dftrain[:,1:]
y_train = dftrain[:,0]

dftest = pd.read_csv(r'E:\pycharm\LR\credit-data-testset.csv')
dftest = np.array(dftest)
X_test = dftest[:,1:-1]
y_test = dftest[:,0]
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
print(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

print(222)
#print(model.predict_proba(X_test))
y_test_pred = model.predict(X_test)
print("accuracy=",accuracy_score(y_test,y_test_pred))
print("auc=",roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
#print(classification_report(y_test, y_test_pred))
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