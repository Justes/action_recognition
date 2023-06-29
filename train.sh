#!/bin/bash

python main.py \
-a r2p1d_18 \
-s hmdb51 \
--frame 8 \
--max-epoch 1 \
--pretrained-model ../pretrained/r2p1d18_K_200ep.pth \
--save-dir logs