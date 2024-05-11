import cv2
import os

if __name__ == '__main__':
    # 图片文件夹路径
    image_folder = './test'

    # 视频输出文件名
    video_name = 'output_video.mp4'

    # 获取图片文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # 排序图片文件列表
    images.sort()

    # 读取第一张图片的宽度和高度
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 设置视频编解码器为H.264
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建视频写入对象
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    # 循环处理每张图片并将其写入视频
    for i in range(len(images)):
        image_path = os.path.join(image_folder, images[i])
        frame = cv2.imread(image_path)

        # 调整图片尺寸以适应视频尺寸
        frame = cv2.resize(frame, (width, height))

        # 初始化透明度和位移参数
        alpha = 1.0
        offset_x, offset_y = 0, 0

        # 每个方向的推入帧数
        num_frames_push = 30

        # 左侧推入
        if i % 4 == 0:
            offset_x = int(width * alpha)
        # 右侧推入
        elif i % 4 == 1:
            offset_x = -int(width * alpha)
        # 上方推入
        elif i % 4 == 2:
            offset_y = int(height * alpha)
        # 下方推入
        else:
            offset_y = -int(height * alpha)

        # 依次推入效果
        for _ in range(num_frames_push):
            pushed_frame = frame.copy()
            pushed_frame[max(0, offset_y):min(height, offset_y + height),
            max(0, offset_x):min(width, offset_x + width)] = frame[max(0, -offset_y):min(height, height - offset_y),
                                                             max(0, -offset_x):min(width, width - offset_x)]

            video.write(pushed_frame)

            # 逐渐增加透明度
            alpha -= 1.0 / num_frames_push
            # 更新位移参数
            offset_x, offset_y = 0, 0

    # 释放资源
    cv2.destroyAllWindows()
    video.release()
