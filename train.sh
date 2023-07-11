#!/bin/bash

python main.py \
-a c3d \
-s hmdb51 \
--optim sgd \
--stepsize 4 \
--max-epoch 1 \
--save-dir logs