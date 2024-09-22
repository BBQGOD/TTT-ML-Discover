#!/bin/bash

cd XGBoost_code

for i in 0 #1 2 3 4
do
    python experiment.py \
        -d bank-marketing \
        -lr 1.0 0.1 0.01 \
        -md 3 5 7 \
        -g 0 0.1 0.3 \
        -nthread 16
done
