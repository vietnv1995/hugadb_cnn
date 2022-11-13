import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("all_data_transformed.csv")
from sklearn import preprocessing
label_name = "act"
features = list(df.columns)
features.remove(label_name)
features.remove("file_name")
x = df[features].values #returns a numpy array
y = df[label_name].values
min_max_scaler = preprocessing.StandardScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns=features)
df[label_name] = y
def make_data_set(df):
    label_name = "act"
    nrows = df.shape[0]
    nrow_split = 100
    features = list(df.columns)
    features.remove(label_name)
    n_samples = nrows//nrow_split
    X = np.zeros((n_samples, len(features), nrow_split))
    y = np.zeros(n_samples)
    for i in range(0, n_samples):
        df_tmp = df[i*nrow_split: (i+1)*nrow_split].copy()
        X[i] = df_tmp[features].values.transpose()
        y[i] = df_tmp[label_name].tolist()[0]
    print(X.shape)
    print(y.shape)
    return X, y

X, y = make_data_set(df)
y = y-1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
clf = SVC()
import time
start = time.time()
print("Start train SVM")
clf.fit(X_train, y_train)
print("Train in {}".format(time.time()-start))
y_train_pred = clf.predict(X_train)
print(metrics.accuracy_score(y_train, y_train_pred))
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))