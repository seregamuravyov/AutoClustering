import sklearn.cluster as cl
import sklearn.mixture as mix
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

datasets = [
    ("iris", "")
]


colors = ['red', 'blue', 'darkorange', 'green', 'black', 'pink','sandybrown', 'darkgreen','aliceblue', 'antiquewhite', 'aqua',
          'aquamarine', 'azure', 'beige', 'rosybrown', 'firebrick', 'sienna',
        'paleturquoise', 'deepskyblue', 'navy', 'm', 'olive',
         'blanchedalmond', 'blueviolet', 'bisque', 'purple'
            'brown', 'burlywood', 'cadetblue', 'chartreuse', 'crimson',]

def make_partitions(data):
    X = []
    with open(data, 'r') as f:
        content = f.readlines()
        for x in content:
            row = x.split()
            res = []
            for i in row:
                res.append(float(i))
            X.append(res)

    pca = PCA(n_components = 2)
    X1 = pca.fit_transform(StandardScaler().fit_transform(X))

    res = []


    res.append(cl.SpectralClustering(eigen_solver="arpack", affinity="nearest_neighbors").fit(X1))
    res.append(cl.KMeans().fit(X1))
    res.append(cl.DBSCAN().fit(X1))
    res.append(cl.MiniBatchKMeans().fit(X1))
    res.append(cl.AffinityPropagation().fit(X1))
    res.append(mix.GaussianMixture(n_components = 5, covariance_type='full').fit(X1))
    res.append(cl.AgglomerativeClustering().fit(X1))
    res.append(cl.Birch().fit(X1))
    res.append(cl.MeanShift().fit(X1))



    for i in range(len(res)):
        ax = plt.subplot(2, 5, i + 1)
        for j in range(len(X1)):
            pair = X1[j]
            if hasattr(res[i], 'labels_'):
                plt.scatter([pair[0]], [pair[1]], s=10, color=colors[res[i].labels_[j]])
            else:
                plt.scatter([pair[0]], [pair[1]], s=10, color=colors[res[i].predict(X1)[j]])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.title(str(i + 1))
        plt.tight_layout()
        plt.gcf().set_size_inches(20, 10)
    #plt.show()

    plt.savefig('foo.png', bbox_inches='tight', dpi = 600)

    plt.show()
    #print(X1)

make_partitions("iris.txt")
