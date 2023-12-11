"""
@Author: yfh
@Date:   2022/11/25
"""

import json
import os
import random

import cv2

# root_path = '/data/yfh/datasets/SIXray/20220304/'
# root_path = '/data/yfh/datasets/OPIXray/'
root_path = '/data/yfh/datasets/HiXray/'

id_category = {0: 'Straight Knife', 1: 'Scissors', 2: 'Gun', 3: 'Wrench', 4: 'Pliers', 5: 'Portable_Charger', 6: 'Water', 7: 'Laptop', 8: 'Mobile_Phone', 9: 'Tablet', 10: 'Cosmetic', 11: 'Lighter'}  # 改成自己的类别

# id_category = {0: 'Straight_Knife', 1: 'Scissors', 2: 'Folding_Knife', 3: 'Utility_Knife', 4: 'Multi-tool_Knife', 5: 'Gun', 6: 'Wrench', 7: 'Pliers'}  # 改成自己的类别



# 'Knife', 'Scissors', 'Folding_Knife', 'Utility_Knife', 'Multi-tool_Knife', 'Gun', 'Wrench', 'Pliers'


def visiual():
    # 获取bboxes
    json_file = os.path.join(root_path, 'annotations', 'train_12cls.json')  # 如果想查看验证集，就改这里
    data = json.load(open(json_file, 'r'))
    images = data['images']  # json中的image列表，

    # 读取图片
    for i in images:  # 随机挑选SAMPLE_NUMBER个检测
        # for i in images:                                        # 整个数据集检查
        img = cv2.imread(os.path.join(root_path, 'train',
                                      i['file_name']))  # 改成验证集的话，这里的图片目录也需要改,train2017 -> val2017
        # cv2.imshow('im',img)
        bboxes = []  # 获取每个图片的bboxes
        category_ids = []
        annotations = data['annotations']
        for j in annotations:
            if j['image_id'] == i['id']:
                bboxes.append(j["bbox"])
                category_ids.append(j['category_id'])

        # 生成锚框
        for idx, bbox in enumerate(bboxes):
            left_top = (int(bbox[0]), int(bbox[1]))  # 这里数据集中bbox的含义是，左上角坐标和右下角坐标。
            right_bottom = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))  # 根据不同数据集中bbox的含义，进行修改。
            cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 3)  # 图像，左上角，右下坐标，颜色，粗细
            cv2.putText(img, id_category[category_ids[idx]], left_top, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
            # 画出每个bbox的类别，参数分别是：图片，类别名(str)，坐标，字体，大小，颜色，粗细
        # cv2.imshow('image', img)                                          # 展示图片，
        # cv2.waitKey(1000)
        cv2.imwrite(os.path.join('visiual_hixray_gt_3', i['file_name']), img)  # 或者是保存图片
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    print('—' * 50)
    os.mkdir('visiual_hixray_gt_3')
    visiual()
    print('| visiual completed.')
    print('| saved as ', os.path.join(os.getcwd(), 'visiual_sixray_gt_2'))
    print('—' * 50)