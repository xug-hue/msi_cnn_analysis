import csv
import os
from sklearn.decomposition import PCA
from sklearn import manifold

class DecompositionMethod:
    def save_decompositon_data(self, dataReduced, method=''):
        saved = False
        i = 0
        while not saved:
            file = "MSI/result/HR2MSI/clustering/" + str(i) + "_" + method + ".csv"
            if not os.path.exists(file):
                with open(file, 'w', encoding='utf-8', newline="") as f:
                    for j in range(len(dataReduced)):
                        csv_write = csv.writer(f)
                        csv_write.writerow(dataReduced[j])
                saved = True
            i = i+1

    def decomposite_data_tSNE(self, data, n_components=2):
        t_sne = manifold.TSNE(
            n_components=n_components,
            learning_rate="auto",
            init="random",
        )
        dataReduced = t_sne.fit_transform(data)

        self.save_decompositon_data(self,dataReduced, "tSNE"+str(n_components))
        return dataReduced

    def decomposite_data_PCA(self, data, n_components=1000):
        pca = PCA(
            n_components=n_components,
            #whiten=True,
        )
        pca.fit(data)
        dataReduced = pca.transform(data)
        self.save_decompositon_data(self,dataReduced, 'PCA'+str(n_components))
        return dataReduced

    def decomposite_data_None(self, data, n_components):
        return data