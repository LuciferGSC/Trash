# -*- coding:utf8 -*-
import os


def rename_all_dir(path):
    fileList = os.listdir(path)
    for item in fileList:
        path_temp = path + "/" + item
        rename_one_dir(path_temp, item)


def rename_one_dir(path, name):
    fileList = os.listdir(path)  # 获取文件路径
    total_num = len(fileList)  # 获取文件长度（个数）
    i = 1  # 表示文件的命名是从1开始的

    for item in fileList:
        if item.endswith(('.jpg', '.JPG')):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
            src = os.path.join(os.path.abspath(path), item)
            dst = os.path.join(os.path.abspath(path), name + str(i) + '.jpg')  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
            try:
                os.rename(src, dst)
                # print('converting %s to %s ...' % (src, dst))
                i = i + 1
            except:
                continue
    print('total %d to rename & converted %d jpg' % (total_num, i - 1))


if __name__ == '__main__':
    rename_all_dir("./ori")
