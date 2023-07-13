#!/bin/bash

python vit_main.py -source_names ucf101 -lr 0.005 -epoch 30 -batch_size 8 -num_workers 4 -num_frames 16 -frame_interval 16 -num_class 101 \
	-arch 'vivit' -attention_type 'fact_encoder' -optim_type 'sgd' -lr_schedule 'cosine' \
	-objective 'supervised' -root_dir '/Users/lin/Code/vivitlog' -pretrain_pth ../pretrained/vivit_K400_pretrained.pth -weights_from 'kinetics'