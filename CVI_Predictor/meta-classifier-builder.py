import autosklearn.classification
import sklearn.model_selection as ms
import sklearn.metrics
import pandas as pd
import numpy as np
import pickle

#GD - 4
#OS - 3
#Sil - 2
#CH - 1

y = ['i1', 'i4', 'i1', 'i3', 'i1', 'i3', 'i1', 'i1', 'i4', 'i3', 'i4', 'i1', 'i3', 'i2', 'i1', 'i2', 'i3', 'i4', 'i2', 'i1', 'i1', 'i1', 'i4', 'i2', 'i4', 'i1', 'i3', 'i1', 'i1', 'i2', 'i1', 'i2', 'i2', 'i2', 'i3', 'i2', 'i1', 'i3', 'i1', 'i3', 'i1', 'i4', 'i3', 'i1', 'i2', 'i3', 'i3', 'i2', 'i2', 'i4', 'i3', 'i2', 'i4', 'i3', 'i1', 'i3', 'i1', 'i2', 'i2', 'i1', 'i2', 'i1', 'i3', 'i1', 'i1', 'i3', 'i1', 'i3', 'i2', 'i4', 'i4', 'i3', 'i2', 'i4', 'i3', 'i4', 'i3', 'i4', 'i2', 'i1', 'i2', 'i4', 'i2', 'i2', 'i4', 'i2', 'i4', 'i1', 'i4', 'i3', 'i3', 'i2', 'i1', 'i2', 'i2', 'i2', 'i4', 'i1', 'i4', 'i4', 'i1', 'i4', 'i3', 'i3', 'i2', 'i2', 'i3', 'i3', 'i3', 'i3', 'i4', 'i1', 'i1', 'i1', 'i1', 'i4', 'i1', 'i4', 'i4', 'i1', 'i1', 'i4', 'i1', 'i4', 'i4', 'i3', 'i1', 'i2', 'i1', 'i4', 'i1', 'i1', 'i3', 'i3', 'i1', 'i1', 'i4', 'i3', 'i2', 'i1', 'i2', 'i4', 'i3', 'i1', 'i2', 'i1', 'i4', 'i1', 'i1', 'i1', 'i3', 'i4', 'i1', 'i1', 'i3', 'i1', 'i3', 'i1', 'i4', 'i1', 'i2', 'i2', 'i2', 'i2', 'i1', 'i1', 'i1', 'i1', 'i1', 'i1', 'i4', 'i1', 'i3', 'i2', 'i1', 'i3', 'i2', 'i2', 'i2', 'i1', 'i3', 'i1', 'i1', 'i1', 'i4', 'i4', 'i4', 'i3', 'i1', 'i2', 'i4', 'i2', 'i1', 'i1', 'i3', 'i4', 'i1', 'i3', 'i2', 1]

# y = []

# for i in range(len(cvis_y)):
#     yi = [0, 0, 0, 0]
#     yi[cvis_y[i] - 1] = 1
#     y.append(yi)

data = pd.read_csv('bestMetric.csv', sep=',', header=None)
X = np.array(data, dtype=np.double).tolist()

#loo = ms.LeaveOneOut()
kf = ms.KFold(n_splits=5, shuffle=True)

y_pred, y_true = [], []
k = 0

for train_index, test_index in kf.split(X):
    k += 1
    print(k)
    x_train, y_train = [], []
    x_test, y_test = [], []
    for i in train_index:
        x_train.append(X[i])
        y_train.append(y[i])

    for i in test_index:
        x_test.append(X[i])
        y_test.append(y[i])

    # x_test.append(X[test_index[0]])
    # y_test.append(y[test_index[0]])

    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=100)
    automl.fit(x_train, y_train, metric=autosklearn.metrics.f1_macro)

    with open('cvi-classifier.pkl', 'wb') as fio:
        pickle.dump(automl, fio)

    for i in test_index:
        xx_test= np.array([np.array(xi) for xi in x_test])
        pred = automl.predict(xx_test)
        for k in range(len(pred)):
            y_pred.append(pred[k])
            y_true.append(y_test[k])

print("F1 = " + str(sklearn.metrics.f1_score(y_true, y_pred, average='micro')))

with open('cvi-classifier.pkl', 'rb') as f:
     automl = pickle.load(f)
     x_test = []
     x_test.append(X[1])
     xx_test = np.array([np.array(xi) for xi in x_test])
     pred = automl.predict(xx_test)
     print(pred)
