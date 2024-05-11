#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import random
import shutil
from shutil import copy2


def divided_data(train_proportion=0.9, val_proportion=0.05):
    save_path = './final'
    base_path = './ori'
    xg = '/'
    # 创建 train、valid、test目录
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path_train = save_path + xg + 'train'
    save_path_valid = save_path + xg + 'valid'
    save_path_test = save_path + xg + 'test'
    if not os.path.exists(save_path_train):
        os.mkdir(save_path_train)
    if not os.path.exists(save_path_valid):
        os.mkdir(save_path_valid)
    if not os.path.exists(save_path_test):
        os.mkdir(save_path_test)
    num_k = 0
    # train、valid、test三个目录中分别创建类别目录
    for category in os.listdir(base_path):
        if num_k < 10:
            temp_save_path_train = save_path_train + xg + "0" + str(num_k) + "_" + category
            temp_save_path_valid = save_path_valid + xg + "0" + str(num_k) + "_" + category
            temp_save_path_test = save_path_test + xg + "0" + str(num_k) + "_" + category
        elif num_k < 100:
            temp_save_path_train = save_path_train + xg + str(num_k) + "_" + category
            temp_save_path_valid = save_path_valid + xg + str(num_k) + "_" + category
            temp_save_path_test = save_path_test + xg + str(num_k) + "_" + category
        num_k = num_k + 1
        if not os.path.exists(temp_save_path_train):
            os.mkdir(temp_save_path_train)
        if not os.path.exists(temp_save_path_valid):
            os.mkdir(temp_save_path_valid)
        if not os.path.exists(temp_save_path_test):
            os.mkdir(temp_save_path_test)
        print("正在处理类别" + category + "...")
        trainfiless = os.path.join(base_path, category)
        trainfiles = os.listdir(trainfiless)
        num_train = len(trainfiles)
        index_list = list(range(num_train))
        random.shuffle(index_list)
        num = 0
        trainDir = temp_save_path_train
        validDir = temp_save_path_valid
        testDir = temp_save_path_test
        for i in index_list:
            fileName = os.path.join(base_path + xg + category, trainfiles[i])
            if num < num_train * train_proportion:
                copy2(fileName, trainDir)
            elif num < num_train * (train_proportion + val_proportion):
                copy2(fileName, validDir)
            else:
                copy2(fileName, testDir)
            num += 1


if __name__ == '__main__':
    # 划分数据集，train_proportion为训练集所占的比例，val_proportion为验证集所占的比例，
    # 剩余的为作为测试集合，如果不像验证集，比例设置为0即可。
    divided_data(train_proportion=0.9, val_proportion=0.05)
