<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="20%" alt="DM-INTERPRETABLEDM-logo">
</p>
<p align="center">
    <h1 align="center">DM-INTERPRETABLEDM</h1>
</p>
<p align="center">
    <em>Experiment, Optimize, Understand-Elevate Your Models!</em>
</p>
<p align="center">
	<!-- local repository, no metadata badges. --></p>
<p align="center">
		<em>Built with the tools and technologies:</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=default&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=default&logo=tqdm&logoColor=black" alt="tqdm">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=default&logo=OpenAI&logoColor=white" alt="OpenAI">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
</p>

<br>

##### 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📂 Repository Structure](#-repository-structure)
- [🧩 Modules](#-modules)
- [🚀 Getting Started](#-getting-started)
    - [🔖 Prerequisites](#-prerequisites)
    - [📦 Installation](#-installation)
    - [🤖 Usage](#-usage)
    - [🧪 Tests](#-tests)
- [📌 Project Roadmap](#-project-roadmap)
- [🤝 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

The dm-interpretabledm project is a comprehensive framework designed for facilitating systematic experimentation with interpretable machine learning models, specifically focusing on K-Nearest Neighbors (KNN) and XGBoost algorithms. It provides users with tools for data preprocessing, hyperparameter tuning, and performance evaluation across diverse datasets, enhancing the model training experience while maintaining data integrity. By incorporating flexible configuration management and automated execution scripts, the project significantly streamlines the workflow for researchers and developers, making it easier to explore, validate, and understand the performance of various machine learning models in practical applications.

---

## 👾 Features

<code>❯ REPLACE-ME</code>

---

## 📂 Repository Structure

```sh
└── dm-interpretabledm/
    ├── KNN_FS_LLM_code
    │   ├── args.py
    │   ├── demo.py
    │   ├── experiment.py
    │   ├── experiment_knn.py
    │   └── run_exp.sh
    ├── XGBoost_code
    │   ├── args.py
    │   ├── experiment.py
    │   ├── experiment_ttt.py
    │   └── run_exp.sh
    ├── preprocess
    │   ├── analyze_test_set.py
    │   ├── analyze_training_set.py
    │   ├── bank_marketing_data_label_distribution.png
    │   ├── bank_marketing_data_mean_std_train.png
    │   ├── boston_housing_data_label_distribution.png
    │   ├── boston_housing_data_mean_std_train.png
    │   ├── breast_cancer_elvira_test_label_distribution.png
    │   ├── breast_cancer_elvira_test_mean_std_train.png
    │   ├── breast_cancer_elvira_train_label_distribution.png
    │   ├── breast_cancer_elvira_train_mean_std_train.png
    │   ├── clean_data.py
    │   ├── clean_data_wo_scaling.py
    │   ├── mean_std_bank_marketing_data_train_scaled.png
    │   ├── mean_std_boston_housing_data_train_scaled.png
    │   ├── mean_std_breast_cancer_elvira_data_test_scaled.png
    │   ├── mean_std_breast_cancer_elvira_data_train_scaled.png
    │   ├── preprocess.sh
    │   ├── preprocess_data.py
    │   ├── preprocess_data_wo_scaling.py
    │   ├── visualization.py
    │   ├── visualization_bank_marketing.png
    │   ├── visualization_bank_marketing_cleaned.png
    │   ├── visualization_boston_housing.png
    │   ├── visualization_boston_housing_cleaned.png
    │   ├── visualization_breast_cancer_elvira_test.png
    │   ├── visualization_breast_cancer_elvira_train.png
    │   └── visualization_breast_cancer_elvira_train_cleaned.png
    ├── requirements.txt
    └── rrl-DM_HW
        ├── .gitignore
        ├── LICENSE
        ├── README.md
        ├── appendix
        ├── args.py
        ├── dataset
        ├── experiment.py
        ├── rrl
        └── scripts
```

---

## 🧩 Modules

<details closed><summary>.</summary>

| File | Summary |
| --- | --- |
| [requirements.txt](dm-interpretabledm/requirements.txt) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>XGBoost_code</summary>

| File | Summary |
| --- | --- |
| [run_exp.sh](dm-interpretabledm/XGBoost_code/run_exp.sh) | Facilitate the execution of experiments for the XGBoost model in a structured manner, allowing for extensive hyperparameter tuning and performance evaluation across various datasets. Manage both standard and advanced testing scenarios, ensuring efficient resource utilization through multi-threading and CUDA support for optimal computational performance. |
| [experiment_ttt.py](dm-interpretabledm/XGBoost_code/experiment_ttt.py) | <code>❯ REPLACE-ME</code> |
| [args.py](dm-interpretabledm/XGBoost_code/args.py) | Facilitates user-defined configurations for XGBoost model training by allowing selection of datasets, tasks, learning rates, and other parameters. Enhances flexibility in model experimentation within the broader architecture of the repository, enabling users to tailor machine learning processes based on specific requirements. |
| [experiment.py](dm-interpretabledm/XGBoost_code/experiment.py) | Facilitates the training and evaluation of XGBoost models for classification and regression tasks. It incorporates data preprocessing, hyperparameter tuning via GridSearchCV, and metrics calculation, seamlessly integrating with the overall architecture of the repository to enhance interpretability and performance assessment across various datasets. |

</details>

<details closed><summary>rrl-DM_HW</summary>

| File | Summary |
| --- | --- |
| [args.py](dm-interpretabledm/rrl-DM_HW/args.py) | Facilitates the configuration and management of training parameters for the Reinforcement Learning model within the repository. It enables users to specify key aspects like dataset, task type, batch size, and learning rate, while organizing outputs and ensuring structured logging for efficient experimentation and tracking. |
| [experiment.py](dm-interpretabledm/rrl-DM_HW/experiment.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>rrl-DM_HW.rrl</summary>

| File | Summary |
| --- | --- |
| [components.py](dm-interpretabledm/rrl-DM_HW/rrl/components.py) | K-Nearest Neighbors (KNN) and XGBoost. Each directory contains scripts designed to facilitate the execution and experimentation of machine learning models, with a strong emphasis on parameter management and experimental setups.Key features of this code include:1. **Experimentation FrameworkThe presence of dedicated `experiment.py` files allows for structured experimentation with the respective algorithms, promoting a systematic approach to model evaluation.2. **Argument ConfigurationThe `args.py` files provide a centralized method for managing parameters and settings, ensuring consistency across different experiments and enhancing reproducibility.3. **Demo CapabilitiesThe `demo.py` scripts are intended to showcase the functionality of the algorithms, making it easier for users to understand how to utilize the models effectively.4. **AutomationThe `run_exp.sh` files automate the execution of experiments, streamlining the workflow for users and reducing the overhead of manual execution.5. **Data PreprocessingThe `preprocess` directory contains scripts dedicated to analyzing and cleaning data, which is essential for preparing datasets for interpretation and model training.Overall, this code is integral to the repositorys goal of developing interpretable machine learning models, providing a structured environment for experimentation, argument management, and data handling. |
| [models.py](dm-interpretabledm/rrl-DM_HW/rrl/models.py) | KNN and XGBoost ImplementationsThe repository houses separate directories for K-Nearest Neighbors (KNN) and XGBoost algorithms, each containing scripts for experimentation and argument parsing.-**Data PreprocessingA dedicated preprocessing folder includes scripts for cleaning data and analyzing training and test datasets, ensuring that the models operate on well-prepared inputs.-**Experiments and DemonstrationsThe structure contains multiple experiment scripts and demo files to facilitate user engagement and understanding of how the algorithms can be applied in practice.Overall, this repository aims to provide a comprehensive framework for conducting machine learning experiments focused on interpretability, showcasing the workflows and tools necessary for data analysis and model validation. |
| [utils.py](dm-interpretabledm/rrl-DM_HW/rrl/utils.py) | Facilitates data ingestion and preprocessing by reading CSV files and handling feature encoding. It employs techniques for discretizing and normalizing continuous variables, enabling efficient data preparation for subsequent machine learning experiments within the repository. This enhances the overall pipelines effectiveness in model training and evaluation. |

</details>

<details closed><summary>rrl-DM_HW.scripts</summary>

| File | Summary |
| --- | --- |
| [run_exp.sh](dm-interpretabledm/rrl-DM_HW/scripts/run_exp.sh) | Facilitates the execution of experiments across multiple datasets, employing grid search to optimize hyperparameters while efficiently utilizing available GPU resources. It ensures systematic testing for classification and regression tasks, contributing to the overarching goal of enhancing model performance within the repositorys architecture. |
| [check_best_param.py](dm-interpretabledm/rrl-DM_HW/scripts/check_best_param.py) | Identifies optimal parameters by analyzing performance metrics from five-fold cross-validation results within a specified directory. The script calculates average macro F1 scores and accuracy, ultimately determining and reporting the best performing model directory, enhancing the overall effectiveness of the RRL framework in the repositorys architecture. |
| [proc_data_for_demo.py](dm-interpretabledm/rrl-DM_HW/scripts/proc_data_for_demo.py) | Facilitates data processing for model demonstrations by converting feature and label CSV files into structured.data and.info files. It ensures compatibility between features and labels, identifies the nature of each feature, and prepares necessary datasets for analysis within the overarching architecture of the repository. |
| [downsample_data.py](dm-interpretabledm/rrl-DM_HW/scripts/downsample_data.py) | Facilitates the downsampling of data from specified input datasets, generating reduced-size outputs for efficient analysis. It serves to create manageable datasets by applying defined sampling ratios, while preserving metadata for reference, enhancing the overall data processing and experimentation workflows in the repository. |
| [check_best_param.sh](dm-interpretabledm/rrl-DM_HW/scripts/check_best_param.sh) | Facilitates the evaluation of model performance by executing scripts to check the best parameters for various datasets, enhancing the overall analysis process within the repository. It generates logs that assist in understanding model efficacy, contributing significantly to the repository’s goal of delivering interpretable machine learning solutions. |
| [run_exp_grid.sh](dm-interpretabledm/rrl-DM_HW/scripts/run_exp_grid.sh) | Facilitates grid searching of hyperparameters for experiments using the bank-marketing dataset, optimizing model training via available GPU resources. It orchestrates various configurations of learning rates, temperature parameters, network structures, and weight decays, ensuring efficient resource management and parallel execution of multiple training jobs. |

</details>

<details closed><summary>rrl-DM_HW.dataset</summary>

| File | Summary |
| --- | --- |
| [tic-tac-toe.info](dm-interpretabledm/rrl-DM_HW/dataset/tic-tac-toe.info) | <code>❯ REPLACE-ME</code> |
| [tic-tac-toe.data](dm-interpretabledm/rrl-DM_HW/dataset/tic-tac-toe.data) | Experimentation FrameworkThe code provides essential functionalities for setting up and running experiments, allowing researchers to evaluate the effectiveness of different models and features in a structured manner.2. **Argument HandlingBy incorporating argument management through `args.py`, the code enables users to customize experiment parameters easily, enhancing usability and flexibility.3. **Demo CapabilitiesThe presence of `demo.py` suggests that there are built-in tools for demonstrating the functionality and effectiveness of the KNN model in practical scenarios.4. **Scripting for AutomationThe inclusion of `run_exp.sh` facilitates automated execution of experiments, streamlining the process of testing various configurations without manual intervention.Overall, this code contributes to the repositorys objective of advancing interpretable machine learning by providing tools specifically tailored for KNN, thereby enhancing the understanding and application of this algorithm in real-world datasets. |

</details>

<details closed><summary>preprocess</summary>

| File | Summary |
| --- | --- |
| [clean_data_wo_scaling.py](dm-interpretabledm/preprocess/clean_data_wo_scaling.py) | Facilitates the cleaning and preprocessing of training datasets by removing outliers, imputing missing values, and mapping categorical features to a normalized range. Enhances data quality for subsequent analysis and modeling within the repository, contributing to robust machine learning workflows across various datasets. |
| [analyze_test_set.py](dm-interpretabledm/preprocess/analyze_test_set.py) | Analyzes breast cancer test data by extracting features, checking for missing values, and generating descriptive statistics. Visualizations highlight mean and standard deviation distributions, along with class label distributions, facilitating better understanding of dataset characteristics crucial for subsequent experimental processes within the repositorys architecture. |
| [visualization.py](dm-interpretabledm/preprocess/visualization.py) | Visualizes clustering results by processing and transforming dataset features, including categorical mapping and missing value imputation. Integrates PCA for dimensionality reduction and generates visual representations of both training and testing datasets, enhancing interpretability and insights within the overall architecture of the dm-interpretabledm repository. |
| [clean_data.py](dm-interpretabledm/preprocess/clean_data.py) | Cleansing data for various datasets by handling missing values, encoding categorical features, and removing outliers using Z-score analysis. It visualizes clustering results post-cleaning, ensuring datasets are ready for further analysis while maintaining original data integrity and providing cleaned outputs for downstream tasks in the repository. |
| [preprocess_data_wo_scaling.py](dm-interpretabledm/preprocess/preprocess_data_wo_scaling.py) | Facilitates data preprocessing for various datasets by loading, cleaning, and normalizing features without scaling. It identifies continuous and discrete features, removes columns with all missing values, and generates cleaned training and testing datasets for further analysis, enhancing the overall functionality and usability of the repository. |
| [preprocess_data.py](dm-interpretabledm/preprocess/preprocess_data.py) | <code>❯ REPLACE-ME</code> |
| [analyze_training_set.py](dm-interpretabledm/preprocess/analyze_training_set.py) | Analyzes training datasets by loading, cleaning, and visualizing data characteristics and distributions. It identifies feature types, checks for missing values, and generates descriptive statistics. Visual outputs include mean, standard deviation, and label distribution plots, enhancing understanding of dataset properties within the repository’s preprocessing framework. |
| [preprocess.sh](dm-interpretabledm/preprocess/preprocess.sh) | Facilitates the comprehensive preprocessing of multiple datasets by executing analyses, data cleaning, scaling, and visualization tasks. Logs all operations to ensure traceability while integrating seamlessly with the repository’s architecture, thereby supporting machine learning experiments and enhancing data quality for further analysis. |

</details>

<details closed><summary>KNN_FS_LLM_code</summary>

| File | Summary |
| --- | --- |
| [run_exp.sh](dm-interpretabledm/KNN_FS_LLM_code/run_exp.sh) | Facilitates the execution of various experiments utilizing KNN and one-shot, few-shot, and many-shot learning strategies across multiple datasets, such as bank marketing and breast cancer. Logs results to enable easy analysis, thereby supporting the broader objectives of model evaluation and performance comparison within the repositorys architecture. |
| [experiment_knn.py](dm-interpretabledm/KNN_FS_LLM_code/experiment_knn.py) | <code>❯ REPLACE-ME</code> |
| [demo.py](dm-interpretabledm/KNN_FS_LLM_code/demo.py) | <code>❯ REPLACE-ME</code> |
| [args.py](dm-interpretabledm/KNN_FS_LLM_code/args.py) | Facilitates configuration management for KNN-based experiments within the dm-interpretabledm repository. Critical features include setting dataset parameters, specifying tasks, defining LLM service credentials, adjusting KNN neighbors, and configuring training batch size and threading, thereby enhancing usability and adaptability for different machine learning scenarios. |
| [experiment.py](dm-interpretabledm/KNN_FS_LLM_code/experiment.py) | K-Nearest Neighbors (KNN) and XGBoost. Each algorithms directory contains scripts designed for setting parameters, executing experiments, and running demonstration models. The presence of `args.py` files indicates that the project supports customizable configurations, enhancing flexibility for users.The `experiment.py` files within both algorithm folders serve as a central mechanism for running and evaluating different model configurations, thereby facilitating comparative analysis. Meanwhile, the `run_exp.sh` scripts streamline the execution process, enabling users to quickly initiate experiments.The `preprocess` directory further enriches the repository by providing essential data cleaning and analysis scripts, allowing users to prepare datasets effectively before experimentation. This design not only promotes organized experimentation but also emphasizes data integrity and insights that are critical for informed modeling decisions.Overall, this code file plays a pivotal role within the repositorys architecture by enabling systematic experimentation and preprocessing, making it easier for researchers and developers to explore and validate machine learning models. |

</details>

---

## 🚀 Getting Started

### 🔖 Prerequisites

**Python**: `version x.y.z`

### 📦 Installation

Build the project from source:

1. Clone the dm-interpretabledm repository:
```sh
❯ git clone ./dm-interpretabledm
```

2. Navigate to the project directory:
```sh
❯ cd dm-interpretabledm
```

3. Install the required dependencies:
```sh
❯ pip install -r requirements.txt
```

### 🤖 Usage

To run the project, execute the following command:

```sh
❯ python main.py
```

### 🧪 Tests

Execute the test suite using the following command:

```sh
❯ pytest
```

---

## 📌 Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## 🤝 Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://LOCAL//dm-interpretabledm/issues)**: Submit bugs found or log feature requests for the `dm-interpretabledm` project.
- **[Submit Pull Requests](https://LOCAL//dm-interpretabledm/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://LOCAL//dm-interpretabledm/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone ./dm-interpretabledm
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to LOCAL**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://LOCAL{//dm-interpretabledm/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=/dm-interpretabledm">
   </a>
</p>
</details>

---

## 🎗 License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## 🙌 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
