#!/bin/bash

cd KNN_FS_LLM_code

python experiment.py \
    -d bank-marketing-wo-scaled.downsample \
    -t classification \
    -ak <api_key> \
    -bu <base_url> \
    -k 3 \
    -bs 5 \
    -nthread 16 > log.bank-marketing.downsample.txt

python experiment.py \
    -d boston-housing-wo-scaled \
    -t regression \
    -ak <api_key> \
    -bu <base_url> \
    -k 3 \
    -bs 5 \
    -nthread 16 > log.boston-housing.txt

python experiment.py \
    -d breast-cancer-{}-wo-scaled \
    -t classification-test \
    -ak sk-UOArhyzuKw4Xaiga3e40F22502B44a6c93CaAaC336A3A1F1 \
    -bu http://15.204.101.64:4000/v1 \
    -k 3 \
    -bs 1 \
    -nthread 16 > log.breast-cancer.txt
