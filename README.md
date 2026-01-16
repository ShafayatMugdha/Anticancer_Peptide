# EvoPhys-ACP: Anticancer Peptide Prediction Framework

## Overview
EvoPhys-ACP is a machine learning framework for predicting anticancer peptides (ACPs) using biologically interpretable features. The framework integrates **physicochemical properties** and **evolutionary information (PSSM)** to develop robust models for ACP classification. 

This repository contains the source code, datasets, and scripts required to reproduce the experiments presented in our study.

---

## Features
- Prediction of anticancer peptides using classical ML models (SVM, Random Forest, Gradient Boosting) and deep learning (MLP, CNN, BiLSTM).
- Feature selection using **SVM-RFE** for optimal predictive performance.
- Hybrid modeling combining physicochemical and evolutionary features.
- Reproducible workflow for training, testing, and evaluating models on benchmark datasets (ACP740, ACP240).

---

## Folder Structure

Anticancer_Peptide/
├── dataset/
│ ├── acp240.txt
│ ├── acp740.txt
│ └── physicochemical_combined_clean.csv
├── Physicochemical code/
│ └── main.py
├── bilstm_core_model.py
├── cnn.py
├── fusion_model.py
├── main_model_SVM-RFE.py
├── main_model_SVM-RFE_240.py
└── README.md
