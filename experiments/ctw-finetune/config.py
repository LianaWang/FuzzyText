model = dict(
    type='Text_Seg',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    mask_head=dict(
        type='SingleMaskHead',
        in_channels=256,
        out_channels=128,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_mask=dict(
            type='PyFocalLoss',
            use_sigmoid=True,
            use_guided=True,
            gamma=2.0,
            alpha=-1,
            loss_weight=1),
        loss_gap=dict(
            type='PyFocalLoss',
            use_sigmoid=True,
            use_guided=True,
            gamma=2.0,
            alpha=-1,
            loss_weight=1),
        loss_contour=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    stuff_head=dict(
        type='PanoFpnHead',
        num_stages=4,
        in_channels=256,
        inner_channels=256,
        num_classes=0,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=None,
        up2x_num=0))
train_cfg = dict()
test_cfg = dict(
    rpn=dict(
        nms_across_levels=True,
        nms_pre=1000,
        nms_post=500,
        max_num=500,
        nms_thr=0.7,
        min_bbox_size=0))
dataset_type = 'CTW1500'
data_root = '../../data/CTW1500/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadCurveAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=0),
    dict(type='CurveRandomRotate', limit_angle=10),
    dict(
        type='CurveResize',
        img_scale=[(1024, 640), (1024, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='CurvePad', size_divisor=32),
    dict(type='DefaultCurveFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_mask', 'gt_mask_gap',
            'gt_mask_contour'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='CTW1500',
            ann_file='../../data/CTW1500/ImageSets/Main/train.txt',
            img_prefix='../../data/CTW1500/',
            cache_root='Cache_contour',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(
                    type='LoadCurveAnnotations',
                    with_bbox=True,
                    with_mask=True),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=0),
                dict(type='CurveRandomRotate', limit_angle=10),
                dict(
                    type='CurveResize',
                    img_scale=[(1024, 640), (1024, 800)],
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='CurvePad', size_divisor=32),
                dict(type='DefaultCurveFormatBundle'),
                dict(
                    type='Collect',
                    keys=[
                        'img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_mask',
                        'gt_mask_gap', 'gt_mask_contour'
                    ])
            ])),
    val=dict(
        type='CTW1500',
        ann_file='../../data/CTW1500/ImageSets/Main/test.txt',
        img_prefix='../../data/CTW1500/',
        cache_root='Cache_contour',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        samples_per_gpu=4,
        type='CTW1500',
        ann_file='../../data/CTW1500/ImageSets/Main/test.txt',
        img_prefix='../../data/CTW1500/',
        cache_root='Cache_contour',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer = dict(type='SGD', lr=0.0001, momentum=0.99, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=0,
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=0.1)
checkpoint_config = dict(interval=100)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
total_epochs = 500
evaluation = dict(
    interval=600,
    proposal_thr_list=[
        0.75, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs'
load_from = './work_dirs/art-pretrain-ctw.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 4)

