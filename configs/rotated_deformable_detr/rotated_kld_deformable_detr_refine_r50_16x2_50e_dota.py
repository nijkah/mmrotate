_base_ = 'rotated_kld_deformable_detr_r50_16x2_50e_dota.py'
model = dict(
            bbox_head=dict(
                with_box_refine=True,
                transformer=dict(
                type='DeformableDetrTransformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention', embed_dims=256),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='RotatedDeformableDetrTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256)
                        ],
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                        'ffn', 'norm')))),
                
                ))
