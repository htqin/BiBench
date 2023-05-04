_base_ = [
    '../_base_/models/resnet34.py', '../_base_/datasets/cifar10.py',
    '../_base_/schedules/sgd_1e-1_cosinelr_200e.py', '../_base_/default_runtime.py'
]

model = dict(arch=dict(
    backbone=dict(type='BiBench_ResNet_CIFAR'),
    head=dict(num_classes=10)))