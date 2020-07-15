import json
import glob
import mne
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import numpy as np

filepath = "./eeg-data/download/eeg*.edf"
edf_files = glob.glob(filepath)
annotations = glob.glob('metadata/annot*.csv')
print(edf_files)


def load_dataset(file, annotations):
    raw = mne.io.read_raw_edf(file, preload=True,
                              verbose=0)
    # filter between 0.5 and 12.8hz
    print(str("Number of data records: {}").format(int(raw.n_times)))
    print(str("Seconds of data: {}").format(int(raw.n_times) // 256))
    raw.filter(l_freq=0.5, h_freq=12.8)
    raw.resample(32)
    print(str("Number of data records after resampling: {}").format(int(raw.n_times)))
    df = raw.to_data_frame()
    df.set_index("time", inplace=True, drop=True)

    eeg_id = file.split("eeg")[-1].split(".")[0]  # example: eeg_id for eeg1.edf is 1
    print(str("eeg patient number: {}").format(eeg_id))

    for i, f in enumerate(annotations):
        annot_df = pd.read_csv(f)
        class_labels = annot_df[eeg_id].dropna()
        # if sampling rate is 256 hz  then downsampled to 32hz and data is labeled every second, this should fit the labels to the data?
        class_labels = [val for val in class_labels for _ in range(32)]
        labs = class_labels[:len(df)]
        df['label_' + str(i)] = labs

    return df

for file in edf_files:
    print(file)
    df = load_dataset(file, annotations)
    csv_name = file.replace("edf", "csv").replace("download", "csv")
    df.to_csv(csv_name)
    print(csv_name)