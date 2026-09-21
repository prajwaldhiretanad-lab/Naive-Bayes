# MAGIC Gamma Telescope Classification: Naive Bayes Pipeline

## Overview
This repository contains a professional machine learning pipeline designed to classify high-energy gamma particles using the MAGIC Gamma Telescope dataset (`magic04.data`)[cite: 5]. The core model applies the Naive Bayes algorithm, which predicts classifications based on probabilistic values derived from Bayes' theorem and conditional probability[cite: 5].

## Dataset Characteristics
The dataset consists of 19,020 records across 11 specific features[cite: 5]. 
* **Features:** `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, and `fDist`[cite: 5].
* **Target Variable:** The `class` column contains string labels ('g' and 'h')[cite: 5]. 
* **Target Encoding:** For modeling purposes, the target variable is transformed into a binary format where 'g' is encoded as 1 and 'h' as 0[cite: 5].

## Data Processing Architecture
The preprocessing pipeline is built to stabilize and balance the data for optimal model performance[cite: 5]:
* **Data Splitting:** The data is partitioned into three distinct sets: Training (60%), Validation (20%), and Testing (20%)[cite: 5].
* **Feature Scaling:** All independent variables are standardized using Scikit-Learn's `StandardScaler` to ensure uniform feature contribution[cite: 5].
* **Class Balancing:** To mitigate inherent class imbalances, `RandomOverSampler` from the `imblearn` library is applied exclusively to the training set[cite: 5]. 

## Modeling and Evaluation
* **Gaussian Naive Bayes:** A `GaussianNB` model is instantiated and fitted to the scaled, oversampled training data[cite: 5]. Model performance is evaluated on the validation set using Scikit-Learn's `classification_report`[cite: 5].
* **Alternative Modeling (KNN):** A K-Nearest Neighbors classifier (`KNeighborsClassifier` with `n_neighbors=3`) is also initialized for comparative evaluation[cite: 5].

*(Note: Ensure that target array dimensions meticulously match prediction arrays during evaluation to prevent `ValueError` exceptions related to inconsistent sample sizes[cite: 5].)*
