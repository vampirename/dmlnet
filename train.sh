conda activate mmdet211
python tools/train.py configs/retinanet/retinanet_r101_fpn_1x_coco.py --work-dir /data/yfh/retinanet-sixray/base-r101-8cls-01/ --gpu-ids 1

python tools/train.py configs/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco.py --work-dir /data/yfh/fcos-sixray/base-r101-8cls-01/ --gpu-ids 1

python tools/train.py configs/ssd/ssd300_coco.py --work-dir /data/yfh/ssd-sixray/base-ssd300-8cls-01/ --gpu-ids 1

python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py --work-dir /data/yfh/retinanet-sixray/base-r50-8cls-01/ --gpu-ids 1

python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py --work-dir /data/yfh/fcos-sixray/base-r50-8cls-01/ --gpu-ids 2
