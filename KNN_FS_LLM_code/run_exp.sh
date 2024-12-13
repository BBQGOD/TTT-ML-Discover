#!/bin/bash

cd KNN_FS_LLM_code

# knn

# python experiment_knn.py \
#     -d bank-marketing \
#     -t classification \
#     -k 15 \
#     -nthread 16 > log.bank-marketing.knn15.txt

python experiment_knn.py \
    -d boston-housing \
    -t regression \
    -k 15 \
    -nthread 16 > log.boston-housing.knn15.txt

# python experiment_knn.py \
#     -d breast-cancer-{} \
#     -t classification-test \
#     -k 3 \
#     -nthread 16 > log.breast-cancer.knn3.txt

# # one-shot learning

# python experiment.py \
#     -d bank-marketing-wo-scaled.downsample \
#     -t classification \
#     -ak <api_key> \
#     -bu <base_url> \
#     -k 1 \
#     -bs 5 \
#     -nthread 16 > log.bank-marketing.downsample.1shot.bs5.txt

# python experiment.py \
#     -d boston-housing-wo-scaled \
#     -t regression \
#     -ak <api_key> \
#     -bu <base_url> \
#     -k 1 \
#     -bs 5 \
#     -nthread 16 > log.boston-housing.1shot.bs5.txt

# python experiment.py \
#     -d breast-cancer-{}-wo-scaled \
#     -t classification-test \
#     -ak <api_key> \
#     -bu <base_url> \
#     -k 1 \
#     -bs 1 \
#     -nthread 16 > log.breast-cancer.1shot.bs1.txt

# # few-shot learning

# python experiment.py \
#     -d bank-marketing-wo-scaled.downsample \
#     -t classification \
#     -ak <api_key> \
#     -bu <base_url> \
#     -k 3 \
#     -bs 5 \
#     -nthread 16 > log.bank-marketing.downsample.3shot.bs5.txt

# python experiment.py \
#     -d boston-housing-wo-scaled \
#     -t regression \
#     -ak <api_key> \
#     -bu <base_url> \
#     -k 3 \
#     -bs 5 \
#     -nthread 16 > log.boston-housing.3shot.bs5.txt

# python experiment.py \
#     -d breast-cancer-{}-wo-scaled \
#     -t classification-test \
#     -ak <api_key> \
#     -bu <base_url> \
#     -k 3 \
#     -bs 1 \
#     -nthread 16 > log.breast-cancer.3shot.bs1.txt

# # many-shot learning

# python experiment.py \
#     -d bank-marketing-wo-scaled.downsample \
#     -t classification \
#     -ak <api_key> \
#     -bu <base_url> \
#     -k 16 \
#     -bs 5 \
#     -nthread 16 > log.bank-marketing.downsample.16shot.bs5.txt

# python experiment.py \
#     -d boston-housing-wo-scaled \
#     -t regression \
#     -ak <api_key> \
#     -bu <base_url> \
#     -k 16 \
#     -bs 5 \
#     -nthread 16 > log.boston-housing.16shot.bs5.txt

# python experiment.py \
#     -d breast-cancer-{}-wo-scaled \
#     -t classification-test \
#     -ak <api_key> \
#     -bu <base_url> \
#     -k 16 \
#     -bs 1 \
#     -nthread 16 > log.breast-cancer.16shot.bs1.txt
