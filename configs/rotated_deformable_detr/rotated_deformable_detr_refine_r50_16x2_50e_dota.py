_base_ = 'rotated_deformable_detr_r50_16x2_50e_dota.py'
model = dict(bbox_head=dict(with_box_refine=True))
