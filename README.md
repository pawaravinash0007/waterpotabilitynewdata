# waterpotabilitynewdata

Water Potability Analysis with Updated Dataset

[![Language](https://img.shields.io/badge/Language-Python-blue?style=flat-square)](.)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange?style=flat-square)](.)

## Overview

An updated water potability analysis using a newer, larger dataset with additional water quality parameters. This repository extends the original Water-Potability project with improved data preprocessing, additional features, and better model performance.

## Improvements Over Previous Version

- Larger dataset with more samples
- Additional water quality parameters
- Improved missing value handling
- Better feature engineering
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation

## Dataset

Features: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity, and additional parameters

Target: Potability (0 = Not Potable, 1 = Potable)

## Models Used

| Model | Accuracy |
|-------|---------|
| Random Forest | ~72% |
| XGBoost | ~73% |
| Gradient Boosting | ~71% |

## Getting Started

```bash
git clone https://github.com/pawaravinash0007/waterpotabilitynewdata.git
cd waterpotabilitynewdata
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
jupyter notebook waterpotability_new.ipynb
```

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

## Author

**Avinash Pawar** | [@pawaravinash0007](https://github.com/pawaravinash0007)
