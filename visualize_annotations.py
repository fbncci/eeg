import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#just use one csv as an example
filepath = "./eeg-data/csv/eeg1.csv"

df = pd.read_csv(filepath)

df.set_index("time", inplace=True)

df.index = df.index.astype("timedelta64[ns]")

info_cols = ["time", "label_0", "label_1", "label_2", "Resp Effort-REF", "ECG EKG-REF"]
label_col = ["label_0", "label_1", "label_2"]
data_cols = list(set(df.columns).difference(info_cols))
cols = df[data_cols].columns

#find find where all annoations agree and make new column for label

X = df[data_cols].values
y = df[label_col].values

max_val = []
labels = []

for row in X:
    max_val.append(max(row))

for row in y:
    if sum(row) == 3:
        labels.append(1)
    else:
        labels.append(0)

df['label'] = labels
df['max'] = max_val

fig, axes = plt.subplots(nrows=17, ncols=1, sharex=True, figsize=(15,15))

datalen = 7000*256

rdf = df

rdf['label'][:datalen].plot(ax=axes[0])
rdf["max"][:datalen].plot(ax=axes[1])
df[cols[0]][:datalen].plot(ax=axes[2])
df[cols[1]][:datalen].plot(ax=axes[3])
df[cols[2]][:datalen].plot(ax=axes[4])
df[cols[3]][:datalen].plot(ax=axes[5])
df[cols[4]][:datalen].plot(ax=axes[6])
df[cols[6]][:datalen].plot(ax=axes[7])
df[cols[7]][:datalen].plot(ax=axes[8])
df[cols[8]][:datalen].plot(ax=axes[9])
df[cols[9]][:datalen].plot(ax=axes[10])
df[cols[10]][:datalen].plot(ax=axes[11])
df[cols[11]][:datalen].plot(ax=axes[12])
df[cols[12]][:datalen].plot(ax=axes[13])
rdf.label_0[:datalen].plot(ax=axes[14])
rdf.label_1[:datalen].plot(ax=axes[15])
rdf.label_2[:datalen].plot(ax=axes[16])

fig.legend()

plt.show()