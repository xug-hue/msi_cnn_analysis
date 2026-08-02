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


def get_feature(csv_file_num, BASE_DIR):
    file = "norm-lxml" + str(csv_file_num) + "(" + str(csv_file_num // pixelNumX) \
           + ',' + str(csv_file_num % pixelNumX) + ')' + '.csv'
    my_table = dt.fread(BASE_DIR + file, sep=",", header=False)  # datatable格式读取文件
    ints = my_table.to_numpy()  # datatable格式转np数组，保存int数据
    if ints.shape != (92064, 1):
        print(ints.shape, csv_file_num)
    return ints


def generate_predict_result_csv(MODEL_FILE, RESULT_DIR, BASE_DIR):
    model = load_model(MODEL_FILE)
    print(model.summary())
    # pre_lists = pd.read_csv(PRE_DIR,sep=r",",header=None)
    pre_result_list = []
    pre_classResult_list = []
    for samplei in range(sampleNum):
        pre_data = np.array(get_feature(samplei, BASE_DIR))
        pre_result = model.predict(np.array([pre_data]))  # 0-1概率预测
        #  print(pre_result) [[9.9986470e-01 1.1371366e-04 2.1607673e-05]]
        pre_result_list.append(pre_result[0])  # 可使用.ravel()
        pre_classResult_list.append([np.argmax(pre_result[0]) + 1])
        if samplei % 1000 == 0:
            print(samplei)

    # print(np.array(pre_result_list).shape)  #多个类的置信概率数组
    # print(np.array(pre_classResult_list).shape) #预测所属类数组
    pre_result_list = np.column_stack((np.array(pre_result_list), np.array(pre_classResult_list)))
    # print(np.array(pre_result_list).shape)    #合并后数组shape

    with open(RESULT_DIR, 'w', encoding='utf-8', newline="") as f:
        for samplei in range(len(pre_result_list)):
            csv_write = csv.writer(f)
            csv_write.writerow(pre_result_list[samplei])
    print("predict finish")

if __name__ == "__main__":
    BASE_DIR = "MSI/data/HR2MSINorm/"
    # The normalized data can be downloaded from: https://pan.baidu.com/s/1GnCpK08nGifOkyB9koYF0A?pwd=ugv8
    RESULT_DIR = "MSI/result/HR2MSI/cnn model/MSIbiaozhuNPArray_predictresult_3class_manuallabel.csv"

    pixelNumX = 260
    sampleNum = 34840
    MODEL_FILE = "msi_1dcnn_best_model.39-0.91-HR2MSI.h5"

    # datatable格式读入,预测结果记录分类1或2，模型视野决定每个热力值的作用范围，本案例是400到1000的某一段范围
    modelScope_table = dt.fread('MSI/data/cnnModelScope.csv', sep=",", header=False).to_numpy()

    generate_predict_result_csv(MODEL_FILE, RESULT_DIR, BASE_DIR)