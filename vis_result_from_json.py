"""
@Author: yfh
@Date:   2022/12/6
"""

import json
import os
import cv2


parent_path = '/data/yfh/datasets/SIXray/20220304/'
# parent_path = '/data/yfh/datasets/OPIXray/'
# parent_path = '/data/yfh/datasets/HiXray/'


# os.mkdir('unidet_sixray_vis')

id_category = {0: 'Straight_Knife', 1: 'Scissors', 2: 'Gun', 3: 'Wrench', 4: 'Pliers'} 
# id_category = {0: 'Straight_Knife', 1: 'Scissors', 2: 'Folding_Knife', 3: 'Utility_Knife', 4: 'Multi-tool_Knife'} 
# id_category = {0: 'Portable_Charger', 1: 'Water', 2: 'Laptop', 3: 'Mobile_Phone', 4: 'Tablet', 5: 'Cosmetic', 6: 'Lighter'}
 

json_file = '/home/yfh/UniDet/output/UniDet/Partitioned_SOH_R50_2x/inference_sixray_test/coco_instances_results.json' # 目标检测生成的文件
name_file = os.path.join(parent_path, 'annotations', 'test_single.json')

with open(json_file, 'r') as annos:
    annotations = json.load(annos)

for i in range(len(annotations)):
    annotation = annotations[i]
    image_id = annotation['image_id']

    data = json.load(open(name_file, 'r'))
    images = data['images']  # json中的image列表

    bbox = annotation['bbox'] # (x1, y1, w, h)
    x, y, w, h = bbox
    for i in images:
        if i['id'] == image_id:
            fname = i['file_name']
            break
    image_path = os.path.join(parent_path, 'test', fname) # 记得加上.jpg
    image = cv2.imread(image_path)
    # 参数为(图像，左上角坐标，右下角坐标，边框线条颜色，线条宽度)
    # 注意这里坐标必须为整数，还有一点要注意的是opencv读出的图片通道为BGR，所以选择颜色的时候也要注意
    anno_image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2) 
    cv2.putText(image, id_category[annotation['category_id']], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    # 参数为(显示的图片名称，要显示的图片)  必须加上图片名称，不然会报错
    # cv2.imshow('demo', anno_image)
    # cv2.waitKey(5000)
    cv2.imwrite(os.path.join('unidet_sixray_vis', fname), image)
    # print('---------------------')