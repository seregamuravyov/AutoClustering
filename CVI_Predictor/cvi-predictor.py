import os
import sys
import pandas as pd
import numpy as np
import autosklearn.classification
import pickle
import MetaFeture_Extractor as mfe

if __name__ == '__main__':
    f = open('cvi-classifier.pkl', 'rb')
    #open('cvi-classifier-small.pkl', 'rb')
    automl = pickle.load(f)

    data = pd.read_csv(sys.argv[1], sep=',', header=None)
    X = np.array(data, dtype=np.double)
    meta_features = np.array([np.array(mfe.MetaDescriptionAll(X))])
    # mfe.MetaDescriptionAll(X, small=True)
    pred = automl.predict(meta_features)

    if (pred[0] == 1):
        print('Calinski-Harabasz index')
    elif (pred[0] == 2):
        print('Silhouette index')
    elif (pred[0] == 3):
        print('OS index')
    else:
        print('Generalized Dunn 41 metric')