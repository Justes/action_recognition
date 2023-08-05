#!/bin/bash

python main.py \
-a r2p1d_50 \
-s hmdb51 \
--optim sgd \
--max-epoch 1 \
--pretrained-model ../pretrained/r2p1d50_K_200ep.pth \
--save-dir logs