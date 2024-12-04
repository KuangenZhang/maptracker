## Model structure
* `plugin/configs/maptracker/nuscenes_oldsplit_kuangen/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py`

* Brief model

```python
model = dict(
    type='MapTracker',
    input = {imgs: (B, N, C, H, W)},
    output = {},
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        input = {imgs: (B, N, C, H, W)},
        output = {},
        img_backbone=dict(
            type='ResNet',
            input = {imgs: (BN, C, H, W)},
            output = {},
            ),
        img_neck=dict(
            type='FPN',
            ),
        transformer=dict(
            type='PerceptionTransformer',
            encoder=dict(
                type='BEVFormerEncoder',
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention'),
                        dict(
                            type='SpatialCrossAttention',
                            deformable_attention=dict(
                                type='MSDeformableAttention3D')
                        )
                    ],
                )
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            ),
    ),
    head_cfg=dict(
        type='MapDetectorHead',
        transformer=dict(
            type='MapTransformer',
            encoder=dict(
                type='PlaceHolderEncoder',
            ),
            decoder=dict(
                type='MapTransformerDecoder_new',
                transformerlayers=dict(
                    type='MapTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                        ),
                        dict(
                            type='MultiheadAttention',
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',       
                    ),
                )
            )
        ),
        loss_cls=dict(
            type='FocalLoss',
        ),
        loss_reg=dict(
            type='LinesL1Loss',
        ),
        assigner=dict(
            type='HungarianLinesAssigner',
                cost=dict(
                    type='MapQueriesCost',
                    ),
                ),
        ),
    seg_cfg=dict(
        type='MapSegHead',
        loss_seg=dict(
            type='MaskFocalLoss',
        ),
        loss_dice=dict(
            type='MaskDiceLoss',
        )
    )
)
```

* Detailed model
```python
model = dict(
    type='MapTracker',
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    history_steps=4,
    test_time_history_steps=20,
    mem_select_dist_ranges=[1, 5, 10, 15],
    skip_vector_head=False,
    freeze_bev=False,
    track_fp_aug=False,
    use_memory=True,
    mem_len=4,
    mem_warmup_iters=-1,
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        history_steps=4,
        use_grid_mask=True,
        img_backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
            ),
        img_neck=dict(
            type='FPN',
            in_channels=[128, 256, 512],
            out_channels=bev_embed_dims,
            start_level=0,
            add_extra_convs=True,
            num_outs=num_feat_levels,
            norm_cfg=norm_cfg,
            relu_before_extra_convs=True),
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=bev_embed_dims,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=bev_embed_dims,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=bev_embed_dims,
                                num_points=8,
                                num_levels=num_feat_levels),
                            embed_dims=bev_embed_dims,
                        )
                    ],
                    feedforward_channels=bev_embed_dims*2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=bev_embed_dims//2,
            row_num_embed=bev_h,
            col_num_embed=bev_w,
            ),
    ),
    head_cfg=dict(
        type='MapDetectorHead',
        num_queries=num_queries,
        embed_dims=embed_dims,
        num_classes=num_class,
        in_channels=bev_embed_dims,
        num_points=num_points,
        roi_size=roi_size,
        coord_dim=2,
        different_heads=False,
        predict_refine=False,
        sync_cls_avg_factor=True,
        trans_loss_weight=0.1,
        transformer=dict(
            type='MapTransformer',
            num_feature_levels=1,
            num_points=num_points,
            coord_dim=2,
            encoder=dict(
                type='PlaceHolderEncoder',
                embed_dims=embed_dims,
            ),
            decoder=dict(
                type='MapTransformerDecoder_new',
                num_layers=6,
                prop_add_stage=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MapTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            num_levels=1,
                            num_points=num_points,
                            dropout=0.1,
                        ),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims*2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),        
                    ),
                    feedforward_channels=embed_dims*2,
                    ffn_dropout=0.1,
                    ## an addtional cross attention for vector memory fusion
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            )
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0
        ),
        loss_reg=dict(
            type='LinesL1Loss',
            loss_weight=50.0,
            beta=0.01,
        ),
        assigner=dict(
            type='HungarianLinesAssigner',
                cost=dict(
                    type='MapQueriesCost',
                    cls_cost=dict(type='FocalLossCost', weight=5.0),
                    reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=permute),
                    ),
                ),
        ),
    seg_cfg=dict(
        type='MapSegHead',
        num_classes=num_class,
        in_channels=bev_embed_dims,
        embed_dims=bev_embed_dims,
        bev_size=(bev_w, bev_h),
        canvas_size=canvas_size,
        loss_seg=dict(
            type='MaskFocalLoss',
            use_sigmoid=True,
            loss_weight=10.0,
        ),
        loss_dice=dict(
            type='MaskDiceLoss',
            loss_weight=1.0,
        )
    ),
    model_name='SingleStage'
)
```