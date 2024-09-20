#!/bin/bash

python preprocess/analyze_training_set.py bank_marketing_data/bank-full.csv >> preprocess/log.txt &
python preprocess/analyze_training_set.py boston_housing_data/HousingData.csv >> preprocess/log.txt &
python preprocess/analyze_training_set.py breast_cancer_elvira_data >> preprocess/log.txt &
python preprocess/analyze_test_set.py breast_cancer_elvira_data >> preprocess/log.txt &
wait

python preprocess/preprocess_data.py bank_marketing_data/bank-full.csv >> preprocess/log.txt &
python preprocess/preprocess_data.py boston_housing_data/HousingData.csv >> preprocess/log.txt &
python preprocess/preprocess_data.py breast_cancer_elvira_data >> preprocess/log.txt &
wait

python preprocess/visualization.py preprocess/bank_marketing_data_X_train_scaled.csv preprocess/bank_marketing_data_y_train.csv >> preprocess/log.txt &
python preprocess/visualization.py preprocess/boston_housing_data_X_train_scaled.csv preprocess/boston_housing_data_y_train.csv >> preprocess/log.txt &
python preprocess/visualization.py "preprocess/breast_cancer_elvira_data_X_{}_scaled.csv" "preprocess/breast_cancer_elvira_data_y_{}.csv" >> preprocess/log.txt &
wait

python preprocess/clean_data.py preprocess/bank_marketing_data_X_train_scaled.csv preprocess/bank_marketing_data_y_train.csv 25.0 >> preprocess/log.txt &
python preprocess/clean_data.py preprocess/boston_housing_data_X_train_scaled.csv preprocess/boston_housing_data_y_train.csv 8.0 >> preprocess/log.txt &
python preprocess/clean_data.py "preprocess/breast_cancer_elvira_data_X_{}_scaled.csv" "preprocess/breast_cancer_elvira_data_y_{}.csv" 8.6 >> preprocess/log.txt &
wait

python rrl-DM_HW/scripts/proc_data_for_demo.py preprocess/bank_marketing_cleaned_train_data.csv preprocess/bank_marketing_cleaned_train_labels.csv bank-marketing >> preprocess/log.txt &
python rrl-DM_HW/scripts/proc_data_for_demo.py preprocess/boston_housing_cleaned_train_data.csv preprocess/boston_housing_cleaned_train_labels.csv boston-housing >> preprocess/log.txt &
python rrl-DM_HW/scripts/proc_data_for_demo.py preprocess/breast_cancer_elvira_cleaned_train_data.csv preprocess/breast_cancer_elvira_cleaned_train_labels.csv breast-cancer-train >> preprocess/log.txt &
python rrl-DM_HW/scripts/proc_data_for_demo.py preprocess/breast_cancer_elvira_data_X_test_scaled.csv preprocess/breast_cancer_elvira_data_y_test.csv breast-cancer-test >> preprocess/log.txt &
wait
