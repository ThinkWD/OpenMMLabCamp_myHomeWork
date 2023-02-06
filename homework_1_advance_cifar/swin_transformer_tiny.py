task_type = 'homework_1_advance_cifar'
work_dir = f'work_dirs/{task_type}'

imgs_prefix = f'data/{task_type}/cifar10'
# model
num_classes = 10
gpu_batch_size = 64
# misc
load_from = (
    'checkpoints/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
)
resume_from = None

# model setting
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False,
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0),
    ],
    train_cfg=dict(
        augments=[
            dict(type='BatchMixup', alpha=0.8, num_classes=num_classes, prob=0.5),
            dict(type='BatchCutMix', alpha=1.0, num_classes=num_classes, prob=0.5),
        ]
    ),
)

# dataset settings
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575], std=[51.5865, 50.847, 51.255], to_rgb=False
)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='Resize', size=224),
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
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix=imgs_prefix,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_prefix=imgs_prefix,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

# schedules settings
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3, 6, 9])
runner = dict(type='EpochBasedRunner', max_epochs=10)
evaluation = dict(interval=1, metric='accuracy')
checkpoint_config = dict(interval=1)

# runtime settings
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
