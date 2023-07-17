#!/bin/bash

python main.py \
-a r2plus1d_18 \
-s hmdb51 \
--optim sgd \
--max-epoch 1 \
--pretrained-model ../pretrained/r2plus1d_18-91a641e6.pth \
--save-dir logs