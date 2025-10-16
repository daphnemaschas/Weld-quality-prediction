# Weld Quality Prediction

**Team Members:** Ghiles Kemiche, Marie Leduc, Adham Noureldin, Daphné Maschas  
**Date:** 2025-10-16  

---

## Project Overview

This project aims to predict the quality of steel welds using machine learning. Weld quality is critical for industries such as wind turbine manufacturing and pipeline construction, where poor welds can lead to costly failures. Traditionally, weld expertise is transferred from expert to expert, making knowledge extraction from data a high-value goal. By leveraging public weld datasets, we aim to:  

- Extract and standardize expert knowledge.  
- Identify key variables representing weld quality.  
- Apply and compare machine learning models, including semi-supervised approaches, to predict weld quality.  

---

## Dataset

The dataset is publicly available at [Weld Database](https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html).  

- **Format:** CSV, containing material properties, welding parameters, and quality labels.  
- **Preprocessing:**  
  - Missing values handling  
  - Feature normalization  
  - Principal Component Analysis (PCA) for feature exploration  

We use Python's Pandas library to load and manipulate the data.  

---

## Objectives

1. Perform exploratory data analysis to understand the dataset and inform preprocessing.  
2. Identify and understand features predictive of weld quality.  
3. Implement and evaluate multiple ML models (supervised and semi-supervised) to predict weld quality.  
4. Conduct rigorous cross-validation and compare model performance using relevant metrics (RMSE, MAE, accuracy).  
5. Provide actionable recommendations for achieving high weld quality.  

---

## Methodology

### Data Preprocessing

- Normalization of numerical features.  
- Handling missing and inconsistent values.  
- PCA to visualize feature importance and correlations.  

### Machine Learning Approaches

- Supervised models: Random Forest, Gradient Boosting, SVM, Logistic Regression.  
- Semi-supervised learning for partially labeled data.  
- Cross-validation to assess model robustness.  

### Evaluation Metrics

- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- Mean Absolute Percentage Error (MAPE)  
- Classification accuracy for discrete weld quality labels  

---

## Results

- Comparative performance analysis of different models.  
- Feature importance insights and interpretation.  
- Recommendations for industrial weld quality improvement based on model outputs.  

---

## Visualizations

We provide visualizations to illustrate:

- Data distributions and feature correlations.  
- PCA results for feature space exploration.  
- Predicted vs actual weld quality comparisons.  

---

## Usage

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/weld-quality-prediction.git
