from tslearn.svm import TimeSeriesSVC
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report, precision_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pickle

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


def class_balance(y):
    return sum(y)/len(y)

print(class_balance(y_train))
print(class_balance(y_test))

def balance_classes(X, y):
    x_return, y_return = [],[]
    for x,label in zip(X, y):
        if label == 1:
            x_return.append(x)
            y_return.append(label)
    n = len(y_return)
    counter = 0
    for x,label in zip(X, y):
        if label == 0:
            x_return.append(x)
            y_return.append(label)
        if counter >= n:
            return(x_return, y_return)
        counter += 1

X_train,y_train = balance_classes(X_train, y_train)
X_test,y_test = balance_classes(X_test,y_test)

print(class_balance(y_train))
print(class_balance(y_test))

clf = TimeSeriesSVC(kernel="gak", gamma=.1)
clf.fit(X_train, y_train)

print("model trained")

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

#evaluate model
y_pred = clf.predict(X_test)
print(str("Accuracy: {}").format(accuracy_score(y_pred, y_test)))
print(str("Precision: {}").format(precision_score(y_pred, y_test)))
print(str("Recall: {}").format(recall_score(y_pred, y_test)))
print(str("F1-Score: {}").format(f1_score(y_pred, y_test)))
print(confusion_matrix(y_pred, y_test))


# n_classes = len(set(y_train))
#
# plt.figure()
# support_vectors = clf.support_vectors_time_series_()
# for i, cl in enumerate(set(y_train)):
#     plt.subplot(n_classes, 1, i + 1)
#     plt.title("Support vectors for class %d" % cl)
#     for ts in support_vectors[i]:
#         plt.plot(ts.ravel())
#
# plt.tight_layout()
# plt.show()