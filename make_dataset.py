import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import copy
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

#just use one csv as an example
filepath = "./eeg-data/csv/eeg*.csv"
files = glob.glob(filepath)
file = files[0]
for file in sorted(files):
    print(file.split("/")[-1])

# TODO: use 8 second windows
# TODO: bandpass filter at 05-12.8 hz
# TODO: downsample to 32hz

#get column information
df = pd.read_csv(file)

df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace("-ref", "")
df = df.reindex(sorted(df.columns), axis=1)
print(df.columns)

info_cols = ["time", "label_0", "label_1", "label_2", "resp effort", "ecg ekg"]
label_col = ["label_0", "label_1", "label_2"]
data_cols = list(set(df.columns).difference(info_cols))
data_cols_copy = list(set(df.columns).difference(info_cols))
data_cols_copy.extend(['id', 'label'])
cols = df[data_cols].columns

# function to process the csv files and extract the labeled sequences
counter = 0


def process_data(file):
    global counter
    try:
        # load dataframe and set index
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace("-ref", "")
        df = df.reindex(sorted(df.columns), axis=1)
        scaler.partial_fit(df.values)

        #         df.set_index("time", inplace=True)
        #         df.index = df.index.astype("timedelta64[ns]")
        #         print(df.columns)

        # extract numpy arrays
        yt = df[label_col].values

        # find find where all annotations agree and make new column for label
        labels = []
        for row in yt:
            if sum(row) == 3:
                labels.append(1)
            else:
                labels.append(0)
        df['label'] = np.array(labels)

        return df
    except Exception as e:
        print(str("ERROR WITH FILE {}, {}").format(file, e))

for i,f in enumerate(files):
    try:
        print(f)
        df = process_data(f)
        df.to_csv("./eeg-data/train/"+str(i)+".csv")
    except TypeError:
        print("error with file ", f)