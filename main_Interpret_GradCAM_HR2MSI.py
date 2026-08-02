import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datatable as dt
import csv
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras import backend as K
from tensorflow.keras import models
import tensorflow as tf

# The normalized data can be downloaded from: https://pan.baidu.com/s/1GnCpK08nGifOkyB9koYF0A?pwd=ugv8
BASE_DIR = "MSI/data/HR2MSINorm/"

pixelNumX = 260
sampleNum = 34840
MODEL_FILE = "msi_1dcnn_best_model.36-0.99.h5"
RESULT_DIR = "MSI/result/HR2MSI/MSIbiaozhuNPArray-predictresult-1d-3class-clustering.csv"

def get_feature(csv_file_num, BASE_DIR=BASE_DIR):
    file = "norm-lxml" + str(csv_file_num) + "(" + str(csv_file_num // pixelNumX) \
           + ',' + str(csv_file_num % pixelNumX) + ')' + '.csv'
    my_table = dt.fread(BASE_DIR + file, sep=",", header=False)  # datatable格式读取文件
    ints = my_table.to_numpy()  # datatable格式转np数组，保存int数据
    if ints.shape != (92064, 1):
        print(ints.shape, csv_file_num)
    return ints

def generate_predict_result_csv(MODEL_FILE=MODEL_FILE, RESULT_DIR=RESULT_DIR):
    model = load_model(MODEL_FILE)
    print(model.summary())
    # pre_lists = pd.read_csv(PRE_DIR,sep=r",",header=None)
    pre_result_list = []
    pre_classResult_list = []
    for samplei in range(sampleNum):
        pre_data = np.array(get_feature(samplei))
        pre_result = model.predict(np.array([pre_data]))  # 0-1概率预测
        #  print(pre_result) [[9.9986470e-01 1.1371366e-04 2.1607673e-05]]
        pre_result_list.append(pre_result[0])  #可使用.ravel()
        pre_classResult_list.append([np.argmax(pre_result[0])+1])
        if samplei%1000==0:
            print(samplei)

    pre_result_list = np.column_stack((np.array(pre_result_list), np.array(pre_classResult_list)))

    with open(RESULT_DIR, 'w', encoding='utf-8', newline="") as f:
        for samplei in range(len(pre_result_list)):
            csv_write = csv.writer(f)
            csv_write.writerow(pre_result_list[samplei])
    print("predict finish")

def generate_interpretable_heatmap_tf2(csv_file_num, model_file=MODEL_FILE):
    model = load_model(model_file)
    pre_data = np.array([get_feature(csv_file_num)])
    # pre_result = model.predict(pre_data)
    conv_layer = model.get_layer('conv1d_7')
    heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(pre_data)
        loss = predictions[:, np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    # 每个样本的热力图计算,并作归一化
    heatmap_nonorm = heatmap.reshape(180, 1).copy()
    heatmapInEachSample[:, csv_file_num] += heatmap_nonorm[:, 0]/np.max(heatmap_nonorm[:, 0])

    # 将每个样本热力图反映到具体质谱特征数据峰上，并对所有样本按类别分开叠加求和
    boolint_feature_data = (np.array(get_feature(csv_file_num)[:, 0]) > 0.0).astype(int)

    for i in range(len(modelScope_table)):
        ileft = modelScope_table[i][0] - 1
        iright = modelScope_table[i][1]

        # 每个样本的热力图按视野叠加计算
        peakHeatmapInEachSample[ileft:iright, csv_file_num] = \
            peakHeatmapInEachSample[ileft:iright, csv_file_num] + \
            boolint_feature_data[ileft:iright] * heatmap_nonorm[i][0]

        # 所有样本按类别分开叠加求和计算
        peakHeatmapInAllSample_nonorm[ileft:iright, predictresult_table[csv_file_num, 0] - 1] = \
            peakHeatmapInAllSample_nonorm[ileft:iright, predictresult_table[csv_file_num, 0] - 1] + \
            boolint_feature_data[ileft:iright] * heatmap_nonorm[i][0]

    # 每个样本的热力图归一化处理
    peakHeatmapInEachSample[:, csv_file_num] = \
        peakHeatmapInEachSample[:, csv_file_num] / np.max(peakHeatmapInEachSample[:, csv_file_num])

def save_peakHeatmapInEachSample():
    # 保存热力图反映到质谱数据上的叠加求和数据到文件
    with open("MSI/result/HR2MSI/peakHeatmapInEachSample.csv", 'w', encoding='utf-8', newline="") as f:
        for i in range(len(peakHeatmapInEachSample)):
            csv_write = csv.writer(f)
            csv_write.writerow(peakHeatmapInEachSample[i])

def save_peakHeatmapInAllSample_nonorm():
    # 保存非正则峰热力值叠加
    with open("MSI/result/HR2MSI/peakHeatmapInAllSample_nonorm.csv", 'w', encoding='utf-8', newline="") as f:
        for i in range(len(peakHeatmapInAllSample_nonorm)):
            csv_write = csv.writer(f)
            csv_write.writerow(peakHeatmapInAllSample_nonorm[i])

def save_heatmapInEachSample():
    # 保存分样本热力图数据到文件
    with open("MSI/result/HR2MSI/heatmapInEachSample.csv", 'w', encoding='utf-8', newline="") as f:
        for i in range(len(heatmapInEachSample)):
            csv_write = csv.writer(f)
            csv_write.writerow(heatmapInEachSample[i])

if __name__ == "__main__":

    peakHeatmapInEachSample = np.zeros((92064, sampleNum))  # 每个样本热力图在具体质谱特征数据上求加权值，并对所有样本求和
    peakHeatmapInAllSample_nonorm = np.zeros((92064, 3))  # 非正则情况
    heatmapInEachSample = np.zeros((180, sampleNum))  # 总热力图是一种分类所有热力图的和

    # datatable格式读入,预测结果记录分类1或2，模型视野决定每个热力值的作用范围，本案例是400到1000的某一段范围
    modelScope_table = dt.fread('MSI/data/cnnModelScope.csv', sep=",", header=False).to_numpy()


    # 对全部样本数据做测试，并将每个类的置信度概率和最高分类类标保存至指定目录csv文件，维度(34840,3)
    # generate_predict_result_csv()
    # 读取预测结果，可通过上面的generate_predict_result_csv()方法生成result，然后放入data文件夹
    predictresult_table = \
        dt.fread('MSI/data/MSIbiaozhuNPArray_3class_CNNPrediction.csv', sep=",", header=False).to_numpy()

    for sampleNumi in range(sampleNum):
        generate_interpretable_heatmap_tf2(sampleNumi)
        if sampleNumi%1000 == 0:
            print(sampleNumi)

    # 保存热力图反映到质谱数据上的叠加求和数据到文件,包括单个
    save_peakHeatmapInEachSample()
    save_peakHeatmapInAllSample_nonorm()

    # 保存总热力图数据到文件
    save_heatmapInEachSample()

