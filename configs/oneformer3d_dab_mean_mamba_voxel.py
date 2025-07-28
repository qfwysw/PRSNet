_base_=['mmdet3d::_base_/default_runtime.py']

custom_imports=dict(imports=['oneformer3d'])

num_classes_scannet=3
voxel_size=0.01
blocks=5
num_channels=64
embed_dims=384
num_layers=3
use_box='mean'
model=dict(
    type='InstancePointTransformer',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=6,
    num_channels=num_channels,
    # num_classes_1dataset=num_classes_structured3d,
    # num_classes_2dataset=num_classes_scannet,
    num_classes_1dataset=num_classes_scannet,
    # prefix_1dataset='structured3d', 
    # prefix_2dataset ='scannet',
    prefix_1dataset ='scannet',
    voxel_size=voxel_size,
    min_spatial_shape=128,
    backbone=dict(
        type='PointTransformerV2MambaVoxel',
        in_channels=6,
        num_classes=3,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 2, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        # grid_sizes=(0.03, 0.075, 0.1875, 0.46875),  # x3, x2.5, x2.5, x2.5
        grid_sizes=(0.06, 0.15, 0.375, 0.9375),  # x3, x2.5, x2.5, x2.5
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="map",  # map / interp
        ),
    decoder=dict(
        type='DabDecoder',
        decoder=dict(
        num_layers=num_layers,
        query_dim=6,
        query_scale_type='cond_elewise',
        with_modulated_hw_attn=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=embed_dims,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=False),
            cross_attn_cfg=dict(
                embed_dims=embed_dims,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=True),
            ffn_cfg=dict(
                embed_dims=embed_dims,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='GELU'))),
        return_intermediate=True),
        num_layers=num_layers,
        num_queries_1dataset=400, 
        num_classes_1dataset=num_classes_scannet,
        prefix_1dataset ='scannet',
        in_channels=num_channels,
        d_model=embed_dims,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=False,
        fix_attention=True),
    criterion=dict(
        type='MaskDataCriterion',
        matcher=dict(
            type='HungarianMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(type='MaskBCECost', weight=1.0),
                dict(type='MaskDiceCost', weight=1.0),
                dict(type='BoxCdist', weight=0.5)]),
        loss_weight=[0.5, 1.0, 1.0, 0.5, 0.5, 0.5],
        non_object_weight=0.05,
        # num_classes_1dataset=num_classes_structured3d,
        # num_classes_2dataset=num_classes_scannet,
        num_classes_1dataset=num_classes_scannet,
        fix_dice_loss_weight=True,
        iter_matcher=True),

    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=400, 
        score_thr=0.0, 
        npoint_thr=100, 
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear'))

# structured3d dataset settings
# scannet dataset settings
dataset_type_scannet='WheelSegDataset'
data_root_scannet='data/wheel158/'
data_prefix=dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')


class_names_scannet=(
    'grain', 'leaf', 'steam')
metainfo_scannet=dict(
    classes=class_names_scannet,
    ignore_index=num_classes_scannet)

train_pipeline_scannet=[
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        # with_label_3d=True,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        seg_3d_dtype='np.int32'),
    dict(type='PointSegClassMapping'),
    dict(type='PointInstClassMapping_',
        num_classes=num_classes_scannet),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor1',
        # type='NormalizePointsColor_', 
         color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size),
    dict(
        type='LoadBox3D',
        use_box=use_box),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'elastic_coords',
            'pts_instance_mask', 'boxes'
        ])
]
test_pipeline_scannet=[
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        seg_3d_dtype='np.int32'),
    dict(
        type='LoadBox3D'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                # type='NormalizePointsColor',
                type='NormalizePointsColor1',
                color_mean=[127.5, 127.5, 127.5]),
        ]),
    dict(type='Pack3DDetInputs_', keys=['points'])
]

train_dataloader=dict(
    batch_size=1,
    num_workers=6,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
            type=dataset_type_scannet,
            data_root=data_root_scannet,
            ann_file='scannet_infos_train.pkl',
            data_prefix=data_prefix,
            metainfo=metainfo_scannet,
            pipeline=train_pipeline_scannet,
            test_mode=False))

# val_evaluator=dict(type='InstanceSegMetric_')
val_dataloader=dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_scannet,
        data_root=data_root_scannet,
        ann_file='scannet_infos_val.pkl',
        metainfo=metainfo_scannet,
        data_prefix=data_prefix,
        pipeline=test_pipeline_scannet,
        test_mode=True))
test_dataloader=val_dataloader
sem_mapping=[0, 1, 2]
# val_evaluator=dict(type='InstanceSegMetric_')

class_names=[
    'grain', 'leaf', 'steam']
label2cat={i: name for i, name in enumerate(class_names)}
metric_meta=dict(
    label2cat=label2cat,
    ignore_index=[3],
    classes=class_names,
    dataset_name='S3DIS')

val_evaluator=dict(type='InstanceSegMetric_')
test_evaluator=val_evaluator

optim_wrapper=dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.02),
    clip_grad=dict(max_norm=2, norm_type=2))
    # clip_grad=dict(type='value', clip_value=1000))

param_scheduler=dict(type='PolyLR', begin=0, end=60000, 
                       power=0.9, by_epoch=False)
log_processor=dict(by_epoch=False)

custom_hooks=[dict(type='EmptyCacheHook', after_iter=True)]
default_hooks=dict(checkpoint=dict(by_epoch=False, save_best='all_ap', rule='greater', interval=60000))

train_cfg=dict(
    type='IterBasedTrainLoop',  # Use iter-based training loop
    max_iters=60000,  # Maximum iterations
    val_interval=500)  # Validation interval
val_cfg=dict(type='ValLoop')
test_cfg=dict(type='TestLoop')


vis_backends=[dict(type='LocalVisBackend')]
visualizer=dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')