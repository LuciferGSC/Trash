# 绘制ROC曲线
# main函数上方有注释，注意看注释
import os
import numpy as np
import time
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from keras.utils.np_utils import to_categorical
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 需要根据需求进行更改的参数
    # ————————————————————————————————————————————————————————————————
    # 按顺序写数据集每个类别的名称，用于绘图时的图注
    labels = ['Bamboo_Chopsticks', 'Beverage_Bottle', 'Big_Bones',
              'Broken_Flowerpots_Dishes', 'Carton', 'Cauliflower_Root',
              'Chopping_Block', 'Chrisma', 'Cosmetic_Bottles',
              'Dentiscalprum', 'Disposable_Fast_Food_Boxes',
              'Dry_Cell', 'Eggshell', 'Expired_Drug', 'Express_Paper_Bag',
              'Fishbone', 'Fouling_Plastic', 'Fruit_Peel', 'Fruit_Pulp',
              'Glass', 'Knapsack', 'Leather_Shoes', 'Leftovers',
              'Metal_food_Cans', 'Oil_Barrel', 'Pan', 'Pillow',
              'Plastic_Bowl', 'Plastic_Hanger', 'Plastic_Toys',
              'Plug_Wire', 'Plush_Toys', 'Portable_Battery',
              'Seasoning_Bottle', 'Shampoo_Bottle', 'Snipe',
              'Tea_Residue', 'Used_Clothes', 'Wine_Bottle', 'Zip_top_can']


    # 测试集的位置
    test_dir = '../data/final/test'
    # 需要绘制ROC曲线的模型的位置
    model_path = "../train/EfficientNetV2S_Att_No_Aug_2023-07-16-05-55-04_best.h5"
    # ————————————————————————————————————————————————————————————————

    img_width, img_height = 224, 224
    batch_size = 16

    # 加载模型
    model = load_model(model_path)

    # 加载测试集数据
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False)

    # 预测结果
    y_pred = model.predict(test_generator, steps=len(test_generator), verbose=1)

    # 获取真实标签
    test_labels = test_generator.classes
    num_classes = test_generator.num_classes

    # 将真实标签转化为one-hot编码
    test_labels = to_categorical(test_labels, num_classes)

    # 计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线
    plt.figure()
    lw = 2
    colors = list(mcolors.CSS4_COLORS.keys())  # 为每个类别选择一种颜色
    for i, color in zip(range(num_classes), colors):
        # plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        #          label='ROC curve of {0} (area:{1:0.4f})'
        #                ''.format(labels[i], roc_auc[i]))
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0}'
                       ''.format(labels[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC curves')
    plt.legend(loc="lower right")
    # 获得当前时间时间戳
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    # plt.savefig('./result/roc_' + otherStyleTime + ".jpg")
    plt.show()
