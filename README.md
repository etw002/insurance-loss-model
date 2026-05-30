# insurance-loss-model
# P&C Insurance Claim Frequency Model
**Tools:** R | tidyverse | ggplot2 | xgboost | pROC

---

## Overview

Built a claim frequency model on 595,000+ auto insurance policyholder 
records from Porto Seguro, a major Brazilian insurer. The goal was to 
predict which policyholders are most likely to file a claim — the same 
problem actuaries solve when building loss cost models for P&C pricing.

Three models were developed and compared using Gini coefficient, the 
standard discrimination metric used in actuarial pricing work.

| Model | AUC | Gini |
|---|---|---|
| XGBoost | 0.646 | 0.292 |
| Poisson GLM | 0.632 | 0.264 |
| Logistic GLM | 0.632 | 0.264 |

---

## Data

- **Source:** Porto Seguro Safe Driver Prediction (Kaggle)
- **Size:** 595,212 policyholders, 57 features
- **Target:** Binary claim indicator (1 = claim filed)
- **Claim frequency:** 3.64% — consistent with real auto insurance loss ratios
- **Features:** Anonymized policyholder and vehicle attributes across 
  binary, categorical, and continuous variable types

---

## Methods

### Cleaning & Feature Engineering
- Converted Porto Seguro's `-1` missing value encoding to `NA`
- Dropped `ps_car_03_cat` (69% missing) and `ps_car_05_cat` (45% missing) 
  as imputation would not be credible at that volume
- Median imputation for continuous variables, mode imputation for 
  categorical and binary variables
- Engineered two features: per-row missing value count (proxy for 
  information withholding) and a car value × region risk interaction term
- 80/20 train/validation split with stratified claim rate check

### Modeling
**Logistic GLM** — Binary classification baseline. GLMs are the 
industry standard for P&C rate filings due to their interpretability 
and regulatory acceptance.

**Poisson GLM** — Reframed the target as claim frequency rather than 
binary outcome, consistent with how actuaries decompose loss costs into 
frequency and severity components.

**XGBoost** — Gradient boosted tree model for comparison. Tuned with 
early stopping on validation AUC to prevent overfitting on the 
imbalanced target.

### Evaluation
Models were evaluated using AUC and Gini coefficient. Lift curves, 
calibration plots, and actual vs. expected (A/E) analysis were 
conducted consistent with actuarial pricing diagnostics.

---

## Key Findings

- XGBoost achieved a Gini coefficient of **0.292**, outperforming both 
  GLM baselines (Gini = 0.264)
- The highest-risk decile filed claims at **2.21x the portfolio average**, 
  demonstrating meaningful risk segmentation
- The XGBoost model was well-calibrated — predicted probabilities 
  closely tracked actual claim rates across the score distribution
- A/E analysis identified systematic under-prediction for policyholder 
  categories 2, 3, and 4, suggesting these segments may warrant 
  separate rating factors in a production pricing model
- Poisson and Logistic GLMs produced identical Gini scores, indicating 
  that frequency reframing did not add discrimination on this anonymized 
  dataset — though the Poisson framework remains more appropriate for 
  real labeled insurance data where exposure is known

---

## Actuarial Context

This project mirrors the workflow of a P&C actuarial analyst building 
a univariate and multivariate loss cost model:

- Frequency-severity decomposition (Poisson GLM for frequency)
- GLM as interpretable baseline consistent with rate filing standards
- Lift curve validation of model discrimination
- A/E ratio analysis to identify pricing inadequacy by segment
- Gini coefficient as primary model selection metric

The main limitation of this dataset is feature anonymization — in 
practice, variables like vehicle type, driver age, and territory would 
be explicitly labeled, enabling more targeted feature engineering and 
clearer business interpretation of model coefficients.
