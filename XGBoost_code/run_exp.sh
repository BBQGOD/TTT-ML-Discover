#!/bin/bash

cd XGBoost_code

for i in 0 #1 2 3 4
do
    python experiment.py \
        -d bank-marketing \
        -ki $i
done
