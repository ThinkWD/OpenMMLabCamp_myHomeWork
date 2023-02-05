task_type = 'homework_1_base_flower'
work_dir = f'work_dirs/{task_type}'

data_root = f'data/{task_type}'
train_file = f'{data_root}/train.txt'
test_file = f'{data_root}/test.txt'
val_file = f'{data_root}/test.txt'
imgs_prefix = data_root
# model
num_classes = 5
gpu_batch_size = 8
# misc
load_from = 'checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
resume_from = None

# model setting
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet', depth=18, num_stages=4, out_indices=(3,), style='pytorch'
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=gpu_batch_size,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=imgs_prefix,
        ann_file=train_file,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix=imgs_prefix,
        ann_file=val_file,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_prefix=imgs_prefix,
        ann_file=test_file,
        pipeline=test_pipeline,
    ),
)

# schedules settings
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=1)
runner = dict(type='EpochBasedRunner', max_epochs=10)
evaluation = dict(interval=1, metric='accuracy')
checkpoint_config = dict(interval=1)

# runtime settings
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
