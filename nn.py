from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import callbacks
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from tslearn.barycenters import softdtw_barycenter
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report, precision_score, confusion_matrix
import numpy as np
import pandas as pd

np.random.seed(42)
cols = ['eeg c3', 'eeg c4', 'eeg cz', 'eeg f3', 'eeg f4', 'eeg f7', 'eeg f8', 'eeg fp1', 'eeg fp2', 'eeg fz', 'eeg o1', 'eeg o2', 'eeg p3', 'eeg p4', 'eeg pz', 'eeg t3', 'eeg t4', 'eeg t5', 'eeg t6']
# cols = ['eeg c3', 'eeg c4', 'eeg cz', 'eeg f3', 'eeg f4', 'eeg f7', 'eeg f8', 'eeg fp1', 'eeg fp2', 'eeg fz', 'eeg o1', 'eeg o2', 'eeg p3', 'eeg p4', 'eeg pz', 'eeg t3', 'eeg t4', 'eeg t5', 'eeg t6', 'resp effort', 'ecg ekg']

def splitDataFrameIntoSmaller(df, chunkSize = 256):
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

def extract_features(df):
    #load the data
    X= df[cols].values
    if 0 in df['label'].values:
        y = 1
    else:
        y = 0
    return(X, y)

def print_evaluation(y_sample, labels):
    print(str("Accuracy: {}").format(accuracy_score(y_sample, labels)))
    print(str("Precision: {}").format(precision_score(y_sample, labels)))
    print(str("Recall: {}").format(recall_score(y_sample, labels)))
    print(str("F1-Score: {}").format(f1_score(y_sample, labels)))
    print(confusion_matrix(y_sample, labels))
    print("\n")


def evaluate(X_train_res, y_train, X_test_res, y_test):
    # evaluate training
    print("Training Results:")
    y_pred = model.predict(X_train_res)
    pred_label = []
    for elm in y_pred:
        if elm[0] > 0.5:
            pred_label.append(1)
        else:
            pred_label.append(0)
    print_evaluation(pred_label, y_train)

    # evalulate testing
    print("Testing Results:")
    y_pred = model.predict(X_test_res)
    pred_label = []
    for elm in y_pred:
        if elm[0] > 0.5:
            pred_label.append(1)
        else:
            pred_label.append(0)
    print_evaluation(pred_label, y_test)


def create_dataset(files):
    X = []
    y = []

    for file in files:
        df_raw = pd.read_csv(file)
        df_arr = splitDataFrameIntoSmaller(df_raw, chunkSize=256)
        for df in df_arr:
            try:
                Xr, yr = extract_features(df)
                #     X_scaled = TimeSeriesScalerMinMax().fit_transform(X.T)
                if len(Xr)==256:
                    X.append(Xr)
                    y.append(yr)
            except:
                pass
    return (X, y)


#get all the prepared csv files
import glob
filepath = "./eeg-data/train/*.csv"
files = glob.glob(filepath)
print(files)

X, y = create_dataset(files)

#split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
pca = PCA(n_components=24, whiten=True, svd_solver='arpack', random_state=1)


for elm in X_train:
    scaler.partial_fit(elm)
    scaler.transform(elm)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# #reshape numpy arrays
# X_train_res = np.asarray(X_train).reshape(len(X_train), 256, len(cols))
# y_train = np.asarray(y_train)
# X_test_res = np.asarray(X_test).reshape(len(X_test), 256, len(cols))
# y_test = np.asarray(y_test)

callback_arr = [callbacks.EarlyStopping(monitor='binary_accuracy',
                                        verbose=1,
                                        mode='min',
                                        patience=9)]



model = Sequential()
model.add(Dense(units=10, return_sequences = True, activation='relu', batch_size=1, kernel_initializer='normal', input_shape = (256, len(cols))))
model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])


# model = Sequential()
# model.add(LSTM(50, return_sequences = True, input_shape = (256, len(cols))))
# model.add(Dropout(0.2))
# model.add(LSTM(50, return_sequences = True ))
# model.add(Dropout(0.2))
# model.add(LSTM(50, return_sequences = True ))
# model.add(Dropout(0.2))
# model.add(LSTM(50))
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])


history = model.fit(X_train_res, y_train, validation_data=(X_test_res, y_test), epochs=15, callbacks=callback_arr)
for key in history.history:
    plt.plot(history.history[key], label=key)
plt.legend()
plt.show()

evaluate(X_train_res, y_train, X_test_res, y_test)