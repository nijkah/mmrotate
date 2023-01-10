_base_ = 'rotated_kld_deformable_detr_refine_r50_16x2_50e_dota.py'
model = dict(bbox_head=dict(as_two_stage=True))
