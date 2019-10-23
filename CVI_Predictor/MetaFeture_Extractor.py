import numpy as np
import Metric as metric
import scipy.stats as st
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from Constants import metrics


def makeD(X):
    D = np.empty((0))
    rows, colums = X.shape
    for i in range(0, rows-1):
        for j in range(i, rows-1):
            if (i != j):
                D = np.append(D, metric.euclidian_dist(X[i], X[j]))

    D /= np.max(D, axis=0)
    return D

def md_percentageD(D, start, end):
    counter = 0
    for d in D:
        if start == 0.0:
            if (d >= start) and (d < end):
                counter += 1
        elif end == 1.0:
            if (d > start) and (d <= end):
                counter += 1
        else:
            if (d > start) and (d < end):
                counter += 1

    return counter / np.size(D, axis=0)


def makeZ(D):
    Z = []
    mean = np.mean(D, axis=0)
    std = np.std(D, axis=0)

    for d in D:
        Z.append((d - mean) / std)

    return Z

def md_percentageZ(Z, start, end):
    counter = 0
    for z in Z:
        if end > 3.0:
            if (z >= start):
                counter += 1
        else:
            if (z >= start) and (z < end):
                counter += 1

    return counter / np.size(Z, axis=0)

def evaluate(X, labels, metric_name):
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    m = metric.metric(X, n_clusters, labels, metric_name)
    return m

def make_landmarks(X):

    result = []
    state_of_the_art = ['KMeans(n_clusters=2)', 'KMeans(n_clusters=5)', 'AffinityPropagation()',
                        'AgglomerativeClustering(n_clusters=2)', 'AgglomerativeClustering(n_clusters=5)',
                        'MeanShift()', 'DBSCAN()', 'KMeans(n_clusters=8)']

    for algo in state_of_the_art:
        cls = eval(algo).fit_predict(X)
        for metric_name in metrics:
            result.append(evaluate(X, cls, metric_name))

    return result


def MetaDescriptionAll(X, small=False):
    result = []
    d = makeD(X)
    z = makeZ(d)

    result.append(np.mean(d, axis=0)) #MD1
    result.append(np.var(d, axis=0)) #MD2
    result.append(np.std(d, axis=0)) #MD3
    result.append(st.skew(d)) #MD4
    result.append(st.kurtosis(d)) #MD5

    result.append(md_percentageD(d, 0.0, 0.1))   #MD6
    result.append(md_percentageD(d, 0.1, 0.2))   #MD7
    result.append(md_percentageD(d, 0.2, 0.3))   #MD8
    result.append(md_percentageD(d, 0.3, 0.4))   #MD9
    result.append(md_percentageD(d, 0.4, 0.5))   #MD10
    result.append(md_percentageD(d, 0.5, 0.6))   #MD11
    result.append(md_percentageD(d, 0.6, 0.7))   #MD12
    result.append(md_percentageD(d, 0.7, 0.8))   #MD13
    result.append(md_percentageD(d, 0.8, 0.9))   #MD14
    result.append(md_percentageD(d, 0.9, 1.0))   #MD15

    result.append(md_percentageZ(z, 0.0, 1.0))  #MD16
    result.append(md_percentageZ(z, 1.0, 2.0))  #MD17
    result.append(md_percentageZ(z, 2.0, 3.0))  #MD18
    result.append(md_percentageZ(z, 3.0, 10.0)) #MD19

    if (small == False):
        landmarks = make_landmarks(X)
        for i in range(len(landmarks)):
            result.append(landmarks[i])

    return result
