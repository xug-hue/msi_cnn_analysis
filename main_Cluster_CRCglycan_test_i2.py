import class_Cluster_Method
import class_Decomposition_Method

import pyimzml.ImzMLParser as imzmlp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # 绘图风格
import numpy as np
import datatable as dt
import csv
import os
import matplotlib
from time import time

#
#
#
# 数据加载
#
#
def normalize(ints):
    sumint = np.sum(ints)
    return ints / sumint * 1000000

def load_info_imzml():
    num_sample = 0
    # CRCglycan data can be downloaded at https://www.ebi.ac.uk/pride/archive/projects/PXD021275
    imzml_path = "MSI/data/i2.imzML"
    # ibd_path = "MSI/data/i2.ibd"
    with imzmlp.ImzMLParser(imzml_path) as parser:
        # 所有谱数据坐标信息，只保留两个维度
        coordinatesNPArray = np.array(parser.coordinates)[:, 0:2]

        # e2数据特殊处理,删除文章中图片没有的数据，左右两块
        list_delete = []
        for i in range(len(coordinatesNPArray)):
            if imzml_path.endswith('e2.imzML'):
                if coordinatesNPArray[i][1] < ((coordinatesNPArray[i][0] - 100) / 14 * (-74) + 142) or coordinatesNPArray[i][0]>390:
                    list_delete.append(i)
        coordinatesNPArray = np.delete(coordinatesNPArray, list_delete, axis=0)

        num_sample = coordinatesNPArray.shape[0]
    return num_sample, coordinatesNPArray

def load_data_imzml():
    imzml_path = "MSI/data/i2.imzML"
    # ibd_path = "MSI/data/i2.ibd"

    with imzmlp.ImzMLParser(imzml_path) as parser:

        # 所有谱数据坐标信息，只保留两个维度
        coordinatesNPArray = np.array(parser.coordinates)[:, 0:2]

        # 保存int数据到数据块
        mzs0, ints0 = parser.getspectrum(0)
        intdata = np.zeros((len(coordinatesNPArray), len(ints0)), dtype=float)

        # 初始化最小和最大的raw数据的数据点数量,最大最小的m/z值
        minlen = 100000
        maxlen = 0
        minmz = 10000.0
        maxmz = 0.0

        # 对所有数据遍历，做处理
        for pixelIndex in range(len(coordinatesNPArray)):
            # 读出指定像素的质谱数据
            mzs, ints = parser.getspectrum(pixelIndex)
            intdata[pixelIndex] = normalize(ints)

            if len(mzs) < minlen:
                minlen = len(mzs)
            if len(mzs) > maxlen:
                maxlen = len(mzs)
            if mzs[0] < minmz:
                minmz = mzs[0]
            if mzs[len(mzs) - 1] > maxmz:
                maxmz = mzs[len(mzs) - 1]

        if minlen == maxlen:
            num_feature = minlen
        print("minlen:", minlen, "maxlen:", maxlen, "minmz:", minmz, "maxmz:", maxmz, "numsample:", coordinatesNPArray.shape[0])

    return intdata

def save_data_csv(data, csvdata_path):
    csvdata_path = csvdata_path
    with open(csvdata_path, 'w', encoding='utf-8', newline="") as f:
        csv_write = csv.writer(f)
        for csv_file_num in range(len(data)):
            csv_write.writerow(data[csv_file_num])


#  主程序入口
#  *
#  *
#  *
#  *
#  *
#  *
# 定义降维、聚类模型和读入数据的方式
if __name__ == "__main__":

    csvdata_path = "MSI/data/i2/MSINorm-all-CRC-i2.csv"
    # 先读出样本数量后面创建数组用，读出坐标信息后面作图用
    sampleNum, coordinateXY = load_info_imzml()
    print(sampleNum, coordinateXY)
    featureNum = 0

    cluster = class_Cluster_Method.ClusterMethod
    decomposition = class_Decomposition_Method.DecompositionMethod

    n_components = 2  # 降到多少维度
    decomposition_method = 'decomposite_data_tSNE'
    # 'decomposite_data_tSNE'    n_components <= 3
    # 'decomposite_data_umap'
    # 'decomposite_data_NMF'
    # 'decomposite_data_PCA'
    # 'decomposite_data_None'

    n_clusters = 3   # 聚类预设类数量
    cluster_method = 'model_HC_Agglomerative'
    # 'model_HC_Agglomerative'
    # 'model_DBSCAN'
    # 'model_kmeans'
    # 'model_AffinityPropagation'

    file = "MSI/data/" + decomposition_method + str(n_components) + ".csv"
    numoftimes = 5
    y_Predicted_nparray = np.zeros((sampleNum, numoftimes))
    y_Predicted = np.array((sampleNum,))

    '''
    # 保存数据到csv文件
    datasave = load_data_imzml()
    save_data_csv(datasave, csvdata_path)
    print('save complete')
    '''

    for i in range(numoftimes):
        t0 = time()
        if not os.path.exists(file):
            # 如果之前没做过降维，完成降维并聚类
            # 加载数据
            data = load_data_imzml()
            print(data.shape)
            # 通过getattr实现函数名字符串动态选择函数
            t1 = time()
            dataReduced = getattr(decomposition, decomposition_method)(decomposition, data, n_components)
            # 内存不够转格式
            # dataReduced = dataReduced.astype(np.uint32)
            t2 = time()
            y_Predicted = getattr(cluster, cluster_method)(cluster, dataReduced, n_clusters)
        else:
            # 如果之前做过降维，直接读取降维数据,但每次降维都具有随机性，重复使用降维结果聚类一般不变
            my_table = dt.fread(file, sep=",", header=False)   # datatable格式读取文件
            t1 = time()
            dataReduced = my_table.to_numpy()  # datatable格式转np数组，保存int数据
            t2 = time()
            #print(dataReduced.shape)
            y_Predicted = getattr(cluster, cluster_method)(cluster, dataReduced, n_clusters)
        t3 = time()
        print('time,', t1-t0, ',', t2-t1, ',', t3-t2, ',', i)
        y_Predicted_nparray[:, i] = y_Predicted

        # DBI和紧凑度计算,DB越小越好，spatialcompactness越大越好
        DB = cluster.print_DB_metrics(cluster, data, y_Predicted)
        spatialcompactness = cluster.print_spatialcompactness_metrics(cluster, coordinateXY, y_Predicted)
        print(DB/spatialcompactness)


        # 聚类结果做像素图，保存，显示
        plt.figure(figsize=(8,5), dpi=100)
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去刻度
        plt.yticks([])  # 去刻度
        # 灰度图
        greycolormap = matplotlib.colors.ListedColormap(['#CCCCCC', '#BBBBBB', '#AAAAAA', '#999999', '#888888',
                                                         '#777777', '#666666', '#555555', '#444444', '#333333',
                                                         '#222222', '#111111', '#000000'])
        plt.scatter(coordinateXY[:, 0],
                    coordinateXY[:, 1],
                    c=y_Predicted,
                    s=1,
                    # 散点大小
                    cmap=greycolormap
                    # 散点颜色集
                    )
        plt.savefig("MSI/result/CRCglycan/clustering/i2/grey" + str(i) + "_" + cluster_method + "_" + str(n_clusters) + ".jpg",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=-0.01)


    # 保存聚类结果到文件
    with open("MSI/result/CRCglycan/clustering/i2/clusterbiaozhu.csv", 'w', encoding='utf-8', newline="") as f:
        for i in range(len(y_Predicted_nparray)):
            csv_write = csv.writer(f)
            csv_write.writerow(y_Predicted_nparray[i])

    with open("MSI/result/CRCglycan/clustering/i2/coordinateXY.csv", 'w', encoding='utf-8', newline="") as f:
        for i in range(len(coordinateXY)):
            csv_write = csv.writer(f)
            csv_write.writerow(coordinateXY[i])
