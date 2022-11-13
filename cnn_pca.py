import pandas as pd
# from pyts.image import RecurrencePlot
# from pyts.image import GramianAngularField
# from pyts.image import MarkovTransitionField
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

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

tmp = df[['acc_rf_x', 'acc_rf_y', 'acc_rf_z', 'gyro_rf_x', 'gyro_rf_y',
       'gyro_rf_z', 'acc_rs_x', 'acc_rs_y', 'acc_rs_z', 'gyro_rs_x',
       'gyro_rs_y', 'gyro_rs_z', 'acc_rt_x', 'acc_rt_y', 'acc_rt_z',
       'gyro_rt_x', 'gyro_rt_y', 'gyro_rt_z', 'acc_lf_x', 'acc_lf_y',
       'acc_lf_z', 'gyro_lf_x', 'gyro_lf_y', 'gyro_lf_z', 'acc_ls_x',
       'acc_ls_y', 'acc_ls_z', 'gyro_ls_x', 'gyro_ls_y', 'gyro_ls_z',
       'acc_lt_x', 'acc_lt_y', 'acc_lt_z', 'gyro_lt_x', 'gyro_lt_y',
       'gyro_lt_z']].values

tmp = tmp.reshape(-1, 100, 12, 3)
from sklearn.decomposition import PCA
X_train, X_test, y_train, y_test = train_test_split(tmp, y, test_size=0.3)
X_train = X_train.reshape(-1, 12*3)
X_test = X_test.reshape(-1, 12*3)
pca = PCA(n_components=12)
X_train_transform = pca.fit_transform(X_train)
X_train_transform = X_train_transform.reshape(-1, 100, 12)
X_test_transform = pca.transform(X_test)
X_test_transform = X_test_transform.reshape(-1, 100, 12)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers import GRU, LSTM, Reshape, TimeDistributed
import random

#
# f = open("result.csv", "w")
# f.write("Layer,Train,Test\n")
checks = set([])
def evaluate_model_cnn_lstm(X_train, y_train, X_test, y_test):
    verbose, epochs, batch_size = 0, 15, 64
    #	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 12 # 100, 12, 3
    print(X_train.shape[1], X_train.shape[2])
    model = Sequential()
    layer_config = []
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(224, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(112, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(224, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(176, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
#     model.add(Flatten())
    model.add(Reshape((-1, 176)))
    model.add(LSTM(16, return_sequences=True))
    model.add(Flatten())
#     model.add(LSTM(22))
    units = 16 * random.randint(1, 3)
    model.add(Dense(units, activation='relu'))
    layer_config.append("dense_{}".format(units))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X_test, y_test))
    # evaluate model
    _, train_accuracy = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
    print("Train: ", train_accuracy)
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print("Test: ", accuracy)

    # row = "{},{},{}\n".format(layer_config_str, train_accuracy, accuracy)
    # f.write(row)
    return accuracy


from tensorflow.keras.utils import to_categorical
import traceback
X_train_transform = X_train_transform.reshape(X_train_transform.shape[0], X_train_transform.shape[1], X_train_transform.shape[2], 1)
X_test_transform = X_test_transform.reshape(X_test_transform.shape[0], X_test_transform.shape[1], X_test_transform.shape[2], 1)
y_train_transform = to_categorical(y_train)
y_test_transform = to_categorical(y_test)
evaluate_model_cnn_lstm(X_train_transform, y_train_transform, X_test_transform, y_test_transform)
