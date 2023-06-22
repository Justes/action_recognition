#!/bin/bash

python main.py \
-a r3d \
-s hmdb51 \
--max-epoch 1 \
--pretrained-model ../pretrained/r3d_18-b3b3357e.pth \
--save-dir logs \