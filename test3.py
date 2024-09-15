from catboost.datasets import adult
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
chen, adult = adult()


for i in adult.columns:
    print(f"The column {i}'s dtype is {adult.loc[:,i].dtype}")

object_columns = []
int_columns = []
for i in adult.columns:
    if adult.loc[:,i].dtype == "object":
        object_columns.append(i)
    if adult.loc[:,i].dtype == "int64":
        int_columns.append(i)

for col_name in object_columns:
    values = np.array(adult[col_name])
    print(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    values = values.reshape(len(values), 1)
    onehot_matrix = onehot_encoder.fit_transform(values)
    print(onehot_matrix)
    adult.drop([col_name],axis=1,inplace=True)
    # 在 Dataframe 首列插入one-hot后新列
    for i in range(onehot_matrix.shape[1]):
        adult.insert(0, 'new_'+col_name+"_"+str(i), value=onehot_matrix[:,i])

print(adult)

for col_name in int_columns:
    Scaler = MinMaxScaler(feature_range=(-1, 1))
    col_value = np.array(adult[col_name]).reshape(-1,1)
    new_col = Scaler.fit_transform(col_value)
    adult[col_name] = new_col

print(adult)

Y = (adult.iloc[:,0]==1)
Y = np.array(Y, dtype=int)
adult.drop(["new_income_1","new_income_0"], axis=1, inplace=True)
X = adult.values
np.savez("./Adult_processed_test.npz", X=X, Y=Y)
