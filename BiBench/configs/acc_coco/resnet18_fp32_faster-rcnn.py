_base_ = [
    '../_base_/models/resnet18_faster-rcnn.py', '../_base_/datasets/coco.py',
    '../_base_/schedules/sgd_2e-2_steplr_24e_warmup.py', '../_base_/default_runtime.py'
]
