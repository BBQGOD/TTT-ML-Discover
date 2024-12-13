#!/bin/bash

cd XGBoost_code

# grid search
# python experiment.py \
#     -d bank-marketing \
#     -t classification \
#     -lr 1.0 0.1 0.01 \
#     -md 3 5 7 \
#     -g 0 0.1 0.3 \
#     -nthread 16 > log.txt

# # best param
# python experiment.py \
#     -d bank-marketing \
#     -t classification \
#     -lr 0.1 \
#     -md 7 \
#     -g 0.3 \
#     -nthread 16 > log.bank-marketing.txt

# python experiment.py \
#     -d boston-housing \
#     -t regression \
#     -lr 0.1 \
#     -md 7 \
#     -g 0.3 \
#     -nthread 16 > log.boston-housing.txt

# python experiment.py \
#     -d breast-cancer-{} \
#     -t classification-test \
#     -lr 0.1 \
#     -md 7 \
#     -g 0.3 \
#     -nthread 16 > log.breast-cancer.txt


# python experiment.py \
#     -d bank-marketing.downsample \
#     -t classification \
#     -lr 0.1 \
#     -md 7 \
#     -g 0.3 \
#     -nthread 16 > log.bank-marketing.downsample.txt

# # ttt
# CUDA_VISIBLE_DEVICES=4 \
# python experiment_ttt.py \
#     -d bank-marketing.downsample \
#     -t classification \
#     -lr 0.1 \
#     -md 7 \
#     -k 1000 \
#     -g 0.3 \
#     --device cuda \
#     -nthread 128 > log.bank-marketing.downsample.ttt.txt


# CUDA_VISIBLE_DEVICES=4 \
# python experiment_ttt.py \
#     -d breast-cancer-{} \
#     -t classification-test \
#     -lr 0.1 \
#     -md 7 \
#     -k 50 \
#     -g 0.3 \
#     --device cuda \
#     -nthread 4 > log.breast-cancer.ttt50.txt

# CUDA_VISIBLE_DEVICES=4 \
# python experiment_ttt.py \
#     -d breast-cancer-{} \
#     -t classification-test \
#     -lr 0.1 \
#     -md 7 \
#     -k 25 \
#     -g 0.3 \
#     --device cuda \
#     -nthread 4 > log.breast-cancer.ttt25.txt

# CUDA_VISIBLE_DEVICES=4 \
# python experiment_ttt.py \
#     -d breast-cancer-{} \
#     -t classification-test \
#     -lr 0.1 \
#     -md 7 \
#     -k 12 \
#     -g 0.3 \
#     --device cuda \
#     -nthread 4 > log.breast-cancer.ttt12.txt

# CUDA_VISIBLE_DEVICES=4 \
# python experiment_ttt.py \
#     -d breast-cancer-{} \
#     -t classification-test \
#     -lr 0.1 \
#     -md 7 \
#     -k 3 \
#     -g 0.3 \
#     --device cuda \
#     -nthread 4 > log.breast-cancer.ttt3.txt


# CUDA_VISIBLE_DEVICES=6 \
# python experiment_ttt.py \
#     -d boston-housing \
#     -t regression \
#     -lr 0.1 \
#     -md 7 \
#     -k 128 \
#     -g 0.3 \
#     --device cuda \
#     -nthread 64 > log.boston-housing.ttt100.txt

CUDA_VISIBLE_DEVICES=2 \
python experiment_ttt.py \
    -d boston-housing \
    -t regression \
    -lr 0.1 \
    -md 7 \
    -k 384 \
    -g 0.3 \
    --device cuda \
    -nthread 64 > log.boston-housing.ttt384.txt
