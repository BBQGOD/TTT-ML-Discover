#!/bin/bash

cd XGBoost_code

# grid search
python experiment.py \
    -d bank-marketing \
    -t classification \
    -lr 1.0 0.1 0.01 \
    -md 3 5 7 \
    -g 0 0.1 0.3 \
    -nthread 16 > log.txt

# best param
python experiment.py \
    -d bank-marketing \
    -t classification \
    -lr 0.1 \
    -md 7 \
    -g 0.3 \
    -nthread 16 > log.bank-marketing.txt

python experiment.py \
    -d boston-housing \
    -t regression \
    -lr 0.1 \
    -md 7 \
    -g 0.3 \
    -nthread 16 > log.boston-housing.txt

python experiment.py \
    -d breast-cancer-{}-wo-scaled \
    -t classification-test \
    -lr 0.1 \
    -md 7 \
    -g 0.3 \
    -nthread 16 > log.breast-cancer.txt
