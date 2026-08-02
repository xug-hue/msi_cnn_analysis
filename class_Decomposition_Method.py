import seaborn as sns; sns.set()  # 绘图风格
import csv
import os
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn import manifold
import umap


class DecompositionMethod:
    #
    #
    #
    #
    # 数据降维
    #
    #
    def save_decompositon_data(self, dataReduced, method=''):
        saved = False
        i = 0
        while not saved:
            file = "MSI/result/CRCglycan/clustering/" + str(i) + "_" + method + ".csv"
            if not os.path.exists(file):
                with open(file, 'w', encoding='utf-8', newline="") as f:
                    for j in range(len(dataReduced)):
                        csv_write = csv.writer(f)
                        csv_write.writerow(dataReduced[j])
                saved = True
            i = i+1


    def decomposite_data_umap(self, data, n_components=2):
        reducer = umap.UMAP(
            random_state=0,
            n_components=n_components,
            low_memory=True,
        )
        embedding = reducer.fit_transform(data)
        print(embedding.shape)

        self.save_decompositon_data(self, embedding, "umap"+str(n_components))
        return embedding

    def decomposite_data_tSNE(self, data, n_components=2):
        t_sne = manifold.TSNE(
            n_components=n_components,
            learning_rate="auto",
            init="random",
            # random_state=0,
        )
        dataReduced = t_sne.fit_transform(data)

        self.save_decompositon_data(self, dataReduced, "tSNE"+str(n_components))
        return dataReduced

    def decomposite_autoencoder(self, data, n_components=100):

        import keras

        encoding_dim = n_components
        # data_dim是一个样本的维度
        data_dim = data.shape[1]
        dimratio = int((data_dim/5/encoding_dim)**(1/3))
        print(dimratio*encoding_dim, dimratio*dimratio*encoding_dim, dimratio*dimratio*dimratio*encoding_dim)
        encoder = keras.models.Sequential([
            keras.layers.Dense(dimratio*dimratio*dimratio*encoding_dim, activation='relu'),
            keras.layers.Dense(dimratio*dimratio*encoding_dim, activation='relu'),
            keras.layers.Dense(dimratio*encoding_dim, activation='relu'),
            keras.layers.Dense(encoding_dim)
        ])

        decoder = keras.models.Sequential([
            keras.layers.Dense(dimratio*encoding_dim, activation='relu'),
            keras.layers.Dense(dimratio*dimratio*encoding_dim, activation='relu'),
            keras.layers.Dense(dimratio*dimratio*dimratio*encoding_dim, activation='relu'),
            keras.layers.Dense(data_dim, activation='tanh')
        ])

        AutoEncoder = keras.models.Sequential([
            encoder,
            decoder
        ])
        AutoEncoder.compile(optimizer='adam', loss='mse')
        AutoEncoder.fit(data, data, epochs=100, batch_size=256)

        dataReduced = encoder.predict(data)
        print(dataReduced.shape)

        self.save_decompositon_data(self, dataReduced, 'autoencoder'+str(n_components))
        return dataReduced


    def decomposite_data_NMF(self, data, n_components=100):
        model = NMF(
            n_components=n_components,
            init='random',
            random_state=0,
            max_iter=300
        )
        model.fit(data)
        #print(model.components_.shape)
        dataReduced = model.transform(data)
        #print(dataReduced.shape)

        # 输出前100主成分的方差和方差占比
        #for i in range(100):
        #    print(pca.explained_variance_[i], end=' ')
        #    print(pca.explained_variance_ratio_[i])

        self.save_decompositon_data(self, dataReduced, 'NMF'+str(n_components))
        return dataReduced

    def decomposite_data_PCA(self, data, n_components=1000):
        pca = PCA(
            n_components=n_components,
            #svd_solver="auto",
            #whiten=True,
        )
        pca.fit(data)
        #print(pca.components_.shape)
        dataReduced = pca.transform(data)
        #print(dataReduced.shape, "random_state:", pca.random_state)

        # 输出前100主成分的方差和方差占比
        #for i in range(100):
        #    print(pca.explained_variance_[i], end=' ')
        #    print(pca.explained_variance_ratio_[i])

        self.save_decompositon_data(self, dataReduced, 'PCA'+str(n_components))
        return dataReduced

    def decomposite_data_None(self, data, n_components):
        return data