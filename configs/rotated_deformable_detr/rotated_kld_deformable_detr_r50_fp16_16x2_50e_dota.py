_base_ = ['./rotated_kld_deformable_detr_r50_16x2_50e_dota.py']

fp16 = dict(loss_scale='dynamic')
