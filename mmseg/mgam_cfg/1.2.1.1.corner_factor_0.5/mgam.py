debug=False

# SETTING: dataset
dataset_type = "CQK_2023_Med"
data_root = r"D:\PostGraduate\DL\mgam_CT\data\2023_Med_CQK"
num_positive_img = 1
num_negative_img = 0
minimum_negative_distance = 50

# SETTING: data preprocess
contrast = 1
brightness = 0
(mean, std) = (561.54, 486.59)      # before clip to (0,4095): (10.87, 1138.37)
source_img_shape = (512,512)
norm_method = 'const'               # const:按照训练集分布归一化, inst:实例归一化。仅针对CTImageEnhance方法
data_preprocessor_normalize = False # 是否启用mmseg的data_preprocessor归一化（归一化可以在自定义的CTImageEnhance中进行）
stretch_CornerFactor=0.5
stretch_GlobalFactor=1
stretch_workers=8                   # 自定义的多进程实现, 与dataloader的num_workers无关
if norm_method == 'inst': assert data_preprocessor_normalize == False, \
    "Instance Norm 只能用CTImageEnhance实现，此时mmseg自带的预处理器应当禁用归一化"


# SETTING: neural network
crop_size = (256,256)
batch_size = 8
lr = 1e-4
workers = 8 if not debug else 0
iters = 15000
val_interval = 500 if not debug else 1

# SETTING: Binary Segmentation Mode
SingleChannelMode = True
num_classes = 2
threshold = 0.3 if SingleChannelMode else None
out_channels = 1 if SingleChannelMode else 2
HeadUseSigmoid = True if SingleChannelMode else False
HeadClassWeight = None if SingleChannelMode else [0.1, 1]




# PreProcess and AfterProcess
reverse_stretch = dict(
            CornerFactor=stretch_CornerFactor, 
            GlobalFactor=stretch_GlobalFactor, 
            in_array_shape=crop_size,
            out_array_shape=crop_size,
            direction="in",
            mmseg_stretch_seg_map=False,
            stretch_num_workers=stretch_workers
)
train_pipeline = [
    dict(type="LoadCTImage"),
    dict(type='LoadCTLabel'),
    dict(type='RandomRotate', prob=1, degree=(-90,90), pad_val=0, seg_pad_val=0),
    dict(type='RadialStretchWithResize', 
            CornerFactor=stretch_CornerFactor, 
            GlobalFactor=stretch_GlobalFactor, 
            in_array_shape=source_img_shape,
            out_array_shape=crop_size, 
            direction="out",
            mmseg_stretch_seg_map=True),
    dict(type='CTImgEnhance', 
         contrast=contrast, 
         brightness=brightness, 
         norm_method=norm_method, 
         mean=mean, std=std),	# CT影像 纵隔部位加强
    dict(type='OriShapeOverride', ori_shape=crop_size), # 使mmseg框架在计算IoU时不会返回到原始尺寸
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type="LoadCTImage"),
    dict(type='LoadCTLabel'),
	dict(type="Resize", scale=crop_size, interpolation="bicubic"),
    dict(type='RadialStretchWithResize', 
            CornerFactor=stretch_CornerFactor, 
            GlobalFactor=stretch_GlobalFactor, 
            in_array_shape=crop_size,
            out_array_shape=crop_size, 
            direction="out",
            mmseg_stretch_seg_map=False),
    dict(type='CTImgEnhance', 
         contrast=contrast, 
         brightness=brightness, 
         norm_method=norm_method, 
         mean=mean, std=std),	# CT影像 纵隔部位加强
    dict(type='OriShapeOverride', ori_shape=crop_size), # 使mmseg框架在计算IoU时不会返回到原始尺寸
    dict(type='PackSegInputs')
]



# dataloader
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        PORT_ARGS=dict(
            root=data_root, 
            metadata_ckpt=r"D:\PostGraduate\DL\ClassicDataset\2023_Med_CQK\2023-11-26_14-23-05.pickle",
            split='train',
            pretraining=False,
            num_positive_img=num_positive_img,
            num_negative_img=num_negative_img,
            minimum_negative_distance=minimum_negative_distance,
            ensambled_img_group=False,
        ),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True if workers > 0 else False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, 
        PORT_ARGS=dict(
            root=data_root, 
            metadata_ckpt=r"D:\PostGraduate\DL\ClassicDataset\2023_Med_CQK\2023-11-26_14-23-05.pickle",
            split='val',
            pretraining=False,
            num_positive_img=num_positive_img,
            num_negative_img=num_negative_img,
            minimum_negative_distance=minimum_negative_distance,
            ensambled_img_group=False,
        ),
        pipeline=test_pipeline
    )
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True if workers > 0 else False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, 
        PORT_ARGS=dict(
            root=data_root, 
            metadata_ckpt=r"D:\PostGraduate\DL\ClassicDataset\2023_Med_CQK\2023-11-26_14-23-05.pickle",
            split='test',
            pretraining=False,
            num_positive_img=num_positive_img,
            num_negative_img=num_negative_img,
            minimum_negative_distance=minimum_negative_distance,
            ensambled_img_group=False,
        ),
        pipeline=test_pipeline
    )
)



# evaluation
val_evaluator = dict(type='ReverseStretchIouMetric',
                     iou_metrics=['mIoU','mDice','mFscore'],
                     nan_to_num=0,
                     reverse_stretch=reverse_stretch
                    )
test_evaluator = val_evaluator



# data preprocessor dict
dpd = data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=mean if data_preprocessor_normalize else None,
	std=std if data_preprocessor_normalize else None,
    pad_val=0,
    seg_pad_val=0,
    non_blocking=True,
)



# optimizer and scheduler
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr),
    clip_grad=dict(type='norm', max_norm=1, norm_type=2, error_if_nonfinite=False)
)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-2, begin=0, end=iters*0.1,
        by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=1e-2*lr,
        power=0.6,
        begin=iters*0.2,
        end=iters,
        by_epoch=False)
]



# Task Control
train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook_ReverseStretch',
                       draw=True, interval=10,
                       mean=mean, std=std,
                       reverse_stretch=reverse_stretch
                       ))



# runtime env
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=0.4)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = True
work_dir = './work_dirs/'
tta_model = None

