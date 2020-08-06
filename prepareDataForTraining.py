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
    if 0 in df['label'].values and 1 in df['label'].values:
        return([],[])
    elif 1 in df['label'].values and 0 not in df['label'].values:
        return(X, 1)
    else:
        return(X, 0)


def create_dataset(files):
    X = []
    y = []
    counter  = 0
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
import random
random.seed(0)
filepath = "./eeg-data/train/*.csv"
files = glob.glob(filepath)
random.shuffle(files)
N = int(len(files)/2)
print(len(files))
print(N)
train_files = files[:N]

X_train, y_train = create_dataset(train_files)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)


test_files = files[N:]

X_test, y_test = create_dataset(test_files)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
