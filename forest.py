from tslearn.svm import TimeSeriesSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report, precision_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


def class_balance(y):
    return sum(y)/len(y)

print(class_balance(y_train))
print(class_balance(y_test))

clf = TimeSeriesSVC(kernel="gak", gamma=.1)
clf.fit(X_train, y_train)
print("Correct classification rate:", clf.score(X_test, y_test))



n_classes = len(set(y_train))

plt.figure()
support_vectors = clf.support_vectors_time_series_()
for i, cl in enumerate(set(y_train)):
    plt.subplot(n_classes, 1, i + 1)
    plt.title("Support vectors for class %d" % cl)
    for ts in support_vectors[i]:
        plt.plot(ts.ravel())

plt.tight_layout()
plt.show()