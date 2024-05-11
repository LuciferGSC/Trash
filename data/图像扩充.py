# -*- coding: utf-8 -*-
'''
图像扩充程序
使用说明-需要扩充的数据文件目录如下：
sourcePicture所指文件夹：
    类别1：
        图片1
        图片2
        。。。
        图片N
    类别2：
        图片1
        图片2
        。。。
        图片N
    。。。。
'''
from PIL import ImageEnhance
from PIL import Image
import numpy as np
import os
import os.path
from PIL import Image, ImageOps, ImageFilter
import random
from scipy import misc
import glob


def getPictureList():
    # set up output dir
    if not os.path.exists(augumentPicture):
        os.mkdir(augumentPicture)

    for category in os.listdir(sourcePicture):
        # for category in ["2"]:
        global saveDir
        saveDir = os.path.join(augumentPicture, category)
        # self.outDir = r'./Lables'
        print("正在处理类别" + category + "...")
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        sourceCategoryPath = os.path.join(sourcePicture, category)
        sourceImageList = glob.glob(os.path.join(sourceCategoryPath, '*'))
        for sourceImage in sourceImageList:
            im = Image.open(sourceImage)
            im = im.convert("RGB")
            im = im.resize((500, 500))
            newFileName = os.path.splitext(os.path.basename(sourceImage))[0] + ".jpg"
            picture = os.path.join(saveDir, newFileName)

            im.save(os.path.join(saveDir, newFileName))

            # 按需求进行扩充, 注释代码，就不进行对应的扩充，如果需要改功能则解除注释即可。
            brightnessTransfer(picture)
            # contrastTransfer(picture)
            # sharpnessTransfer(picture)
            rotateTransfer(picture)
            flipTransfer(picture)
            # gaussianTransfer(picture)

    print("处理完成")


def brightnessTransfer(picture):  # 亮度
    image = Image.open(picture)
    image = image.convert("RGB")
    enhancer = ImageEnhance.Brightness(image)

    factor = 0.5
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_brightness_1.jpg"
    enhancedImage = enhancer.enhance(factor)
    enhancedImage = enhancedImage.save(os.path.join(saveDir, newFileName))

    factor = 1.5
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_brightness_2.jpg"
    enhancedImage = enhancer.enhance(factor)
    enhancedImage = enhancedImage.save(os.path.join(saveDir, newFileName))


def contrastTransfer(picture):  # 对比度
    image = Image.open(picture)
    image = image.convert("RGB")
    enhancer = ImageEnhance.Contrast(image)
    factor = 0.5
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_contrast_1.jpg"
    enhancedImage = enhancer.enhance(factor)
    enhancedImage = enhancedImage.save(os.path.join(saveDir, newFileName))

    factor = 1.5
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_contrast_2.jpg"
    enhancedImage = enhancer.enhance(factor)
    enhancedImage = enhancedImage.save(os.path.join(saveDir, newFileName))


def sharpnessTransfer(picture):  # 锐化度
    image = Image.open(picture)
    image = image.convert("RGB")
    enhancer = ImageEnhance.Sharpness(image)
    # image2 = image.rotate(60)
    # for i in range(4):
    #    factor = (i+1) / 2.0
    #    enhancer.enhance(factor).show("Sharpness %f" % factor)
    factor = 0.5
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_sharpness_1.jpg"
    enhancedImage = enhancer.enhance(factor)
    enhancedImage = enhancedImage.save(os.path.join(saveDir, newFileName))

    factor = 1.5
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_sharpness_2.jpg"
    enhancedImage = enhancer.enhance(factor)
    enhancedImage = enhancedImage.save(os.path.join(saveDir, newFileName))


def rotateTransfer(picture):  # 旋转图片
    img = Image.open(picture)
    img = img.convert("RGB")
    # img.show()
    img2 = img.rotate(90)
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_rotate_1.jpg"
    img2.save(os.path.join(saveDir, newFileName))

    img2 = img.rotate(180)
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_rotate_2.jpg"
    img2.save(os.path.join(saveDir, newFileName))

    img2 = img.rotate(270)
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_rotate_3.jpg"
    img2.save(os.path.join(saveDir, newFileName))


def flipTransfer(picture):
    img = Image.open(picture)
    img = img.convert("RGB")
    # img.show()
    x = img.size[0]
    y = img.size[1]
    img = img.load()
    c = Image.new("RGB", (x, y))
    d = Image.new("RGB", (x, y))

    for i in range(0, x):
        for j in range(0, y):
            w = x - i - 1
            h = y - j - 1
            rgb = img[w, j]  # 镜像翻转
            # rgb=img[w,h] #翻转180度
            # rgb=img[i,h] #上下翻转
            rgb2 = img[i, h]  # 上下翻转
            c.putpixel([i, j], rgb)
            d.putpixel([i, j], rgb2)
    # c.show()
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_flip_1.jpg"
    c.save(os.path.join(saveDir, newFileName))

    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_flip_2.jpg"
    d.save(os.path.join(saveDir, newFileName))


def gaussianTransfer(picture):
    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    # 将图像转化成数组
    mean = 0.2
    sigma = 0.3
    image = Image.open(picture)
    image = image.convert("RGB")
    img = np.array(image)
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    saveImage = Image.fromarray(np.uint8(img))
    newFileName = os.path.splitext(os.path.basename(picture))[0] + "_gaussian_1.jpg"

    saveImage.save(os.path.join(saveDir, newFileName))


if __name__ == '__main__':
    root = r'./final'
    saveDir = "./final_Augmentation"
    for category in os.listdir(root):
        print(root+'/'+category)
        sourcePicture = root+'/'+category
        augumentPicture = sourcePicture + '_Augmentation'
        getPictureList()

