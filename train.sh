#!/bin/bash

python main.py \
-s ucf101 \
--max-epoch 1 \
--resume ../pretrained/c3d_ucf101_2023-04-18_1.pth.tar \
--save-dir logs \