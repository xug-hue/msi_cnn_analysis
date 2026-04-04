import numpy as np
import datatable as dt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

class ClusterMethod:
    def model_kmeans(self, data, n_clusters):
        kmeans = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
        )
        kmeans.fit(data)
        labels = kmeans.predict(data)
        return labels

    def model_HC_Agglomerative(self, data, n_clusters):
        aClustering = AgglomerativeClustering(
            affinity='euclidean',
            linkage='ward',
            n_clusters=n_clusters,
        )
        aClustering.fit(data)
        labels = aClustering.labels_
        return labels

    def get_labels_true():
        # manual annotation
        file = "MSI/data/MSIbiaozhuNPArray_3class_manulimproved.csv"
        my_table = dt.fread(file, sep=",", header=False)
        labels_true = my_table.to_numpy()
        return labels_true[:, 0]

    def print_supervised_metrics(self, labels):
        labels_true = self.get_labels_true()
        # Adjusted Rand Index
        print("ARI, %0.4f," % metrics.adjusted_rand_score(labels_true, labels), end='')

    def print_DB_metrics(self, data, labels):
        davies_bouldin_score = metrics.davies_bouldin_score(data, labels)
        print("davies_bouldin_score: %0.3f" % davies_bouldin_score)
        return davies_bouldin_score

    def print_spatialcompactness_metrics(self, coordinate, labels):
        import csv
        if len(labels) != len(coordinate):
            print('data error')
        sameClusterRatio = np.zeros(len(labels), dtype=float)
        for i in range(len(labels)):
            adjacentPixelNum = 0.0
            sameAdjacentPixelNum = 0.0
            leftupi = np.where(((coordinate[i] - [1, -1]) == coordinate).all(1))[0]
            if len(leftupi) == 1:
                adjacentPixelNum += 1
                if labels[leftupi[0]] == labels[i]:
                    sameAdjacentPixelNum += 1
            upi = np.where(((coordinate[i] - [0, -1]) == coordinate).all(1))[0]
            if len(upi) == 1:
                adjacentPixelNum += 1
                if labels[upi[0]] == labels[i]:
                    sameAdjacentPixelNum += 1
            rightupi = np.where(((coordinate[i] - [-1, -1]) == coordinate).all(1))[0]
            if len(rightupi) == 1:
                adjacentPixelNum += 1
                if labels[rightupi[0]] == labels[i]:
                    sameAdjacentPixelNum += 1
            lefti = np.where(((coordinate[i] - [1, 0]) == coordinate).all(1))[0]
            if len(lefti) == 1:
                adjacentPixelNum += 1
                if labels[lefti[0]] == labels[i]:
                    sameAdjacentPixelNum += 1
            righti = np.where(((coordinate[i] - [-1, 0]) == coordinate).all(1))[0]
            if len(righti) == 1:
                adjacentPixelNum += 1
                if labels[righti[0]] == labels[i]:
                    sameAdjacentPixelNum += 1
            leftdowni = np.where(((coordinate[i] - [1, 1]) == coordinate).all(1))[0]
            if len(leftdowni) == 1:
                adjacentPixelNum += 1
                if labels[leftdowni[0]] == labels[i]:
                    sameAdjacentPixelNum += 1
            downi = np.where(((coordinate[i] - [0, 1]) == coordinate).all(1))[0]
            if len(downi) == 1:
                adjacentPixelNum += 1
                if labels[downi[0]] == labels[i]:
                    sameAdjacentPixelNum += 1
            rightdowni = np.where(((coordinate[i] - [-1, 1]) == coordinate).all(1))[0]
            if len(rightdowni) == 1:
                adjacentPixelNum += 1
                if labels[rightdowni[0]] == labels[i]:
                    sameAdjacentPixelNum += 1
            if adjacentPixelNum != 0:
                sameClusterRatio[i] = sameAdjacentPixelNum / adjacentPixelNum
            else:
                sameClusterRatio[i] = 0
        with open("MSI/data/spatialcompactness_metrics.csv", 'w', encoding='utf-8', newline="") as f:
            for i in range(len(sameClusterRatio)):
                csv_write = csv.writer(f)
                csv_write.writerow([sameClusterRatio[i]])
        averageClusterRatio = np.zeros(max(labels)+1)
        for i in range(max(labels)+1):
            averageClusterRatio[i] = sum(sameClusterRatio[labels==i])/len(sameClusterRatio[labels==i])
        print(np.around(averageClusterRatio, 3), np.around(np.mean(averageClusterRatio), 3))
        return np.mean(averageClusterRatio)

    def DB_metrics_info(self, data, labels):
        from sklearn.utils import check_X_y
        from sklearn.utils import _safe_indexing
        from sklearn.metrics.pairwise import pairwise_distances
        from sklearn.preprocessing import LabelEncoder
        X, labels = check_X_y(data, labels)
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        n_samples, _ = X.shape
        n_labels = len(le.classes_)
        intra_dists = np.zeros(n_labels)
        centroids = np.zeros((n_labels, len(X[0])), dtype=float)
        for k in range(n_labels):
            cluster_k = _safe_indexing(X, labels == k)
            centroid = cluster_k.mean(axis=0)
            centroids[k] = centroid
            intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))
        centroid_distances = pairwise_distances(centroids)
        if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
            return 0.0
        centroid_distances[centroid_distances == 0] = np.inf
        combined_intra_dists = intra_dists[:, None] + intra_dists
        scores = np.max(combined_intra_dists / centroid_distances, axis=1)
        return scores, combined_intra_dists, centroid_distances