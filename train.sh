#!/bin/bash

python main.py \
-a r2d_18 \
-s hmdb51 \
--frame 1 \
--optim sgd \
--max-epoch 1 \
--pretrained-model ../pretrained/resnet18-f37072fd.pth \
--save-dir logs