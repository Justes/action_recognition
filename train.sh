#!/bin/bash

python main.py \
-s hmdb51 \
--max-epoch 1 \
--pretrained-model ../pretrained/c3d-pretrained.pth \
--save-dir logs \