# 项目介绍：

```
D:.
└─Trash
    ├─data			存放原始数据，数据集处理代码和数据说明文件
    ├─train			存放训练具有识别功能的卷积神经网络的脚本以及训练日志等
    ├─analyse		存放模型预测、可视化等对模型进行的分析的功能脚本
    ├─classify		存放网站的后端文件
    ├─classifyclass		存放网站的后端文件
    ├─media				存放上传的待测试图片
    ├─static			存放网站的静态资源，包括css，背景图片等等
    ├─templates			存放前端的html页面
    ├─pretrained		存放预训练模型
    ├─images			存放GUI客户端程序使用到的素材图片
    ├─manage.py			Django框架搭建的网站的入口文件
    └─window.py			GUI客户端程序的入口文件
```

注意力机制、准确率等、图像扩充

命令备份：
进入项目根目录，输入cmd，打开命令行，输入下列命令，执行不同的功能。
tensorboard启动：
tensorboard --logdir=logs文件夹所在的绝对路径
Django网站启动：
python manage.py runserver
或者
python manage.py runserver 127.0.0.1:XXXX
xxxx可以填任意数字，表示在不同的端口中打开Django项目
启动GUI客户端程序：
python window.py



