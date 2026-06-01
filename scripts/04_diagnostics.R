# ============================================================
# 04_diagnostics.R
# Porto Seguro - Model Diagnostics & Experience Analysis
# ============================================================

library(tidyverse)
library(ggplot2)
library(patchwork)
library(pROC)

# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

val_preds   <- read_csv("data/val_predictions.csv")
val_final   <- read_csv("data/val_final.csv")
model_results <- read_csv("data/model_results.csv")

# Rejoin val_final so we have the original features for subgroup analysis
val_full <- val_final %>%
  bind_cols(val_preds %>% select(xgb_pred, logistic_pred, poisson_pred))


# ============================================================
# 2. ROC CURVES — All 3 models on one plot
# Visual confirmation of Gini scores
# ============================================================

roc_xgb      <- roc(val_preds$actual, val_preds$xgb_pred)
roc_logistic <- roc(val_preds$actual, val_preds$logistic_pred)
roc_poisson  <- roc(val_preds$actual, val_preds$poisson_pred)

# Build a plottable dataframe from each ROC object
roc_to_df <- function(roc_obj, model_name) {
  tibble(
    fpr   = 1 - roc_obj$specificities,
    tpr   = roc_obj$sensitivities,
    model = model_name
  )
}

roc_df <- bind_rows(
  roc_to_df(roc_xgb,      "XGBoost (Gini = 0.292)"),
  roc_to_df(roc_logistic, "Logistic GLM (Gini = 0.264)"),
  roc_to_df(roc_poisson,  "Poisson GLM (Gini = 0.264)")
)

ggplot(roc_df, aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              color = "grey50") +
  scale_color_manual(values = c("#D7191C", "#2C7BB6", "#ABD9E9")) +
  labs(
    title    = "ROC Curves — Model Comparison",
    subtitle = "Dashed line = random classifier baseline",
    x        = "False Positive Rate",
    y        = "True Positive Rate",
    color    = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/10_roc_curves.png", width = 7, height = 6)


# ============================================================
# 3. PREDICTED SCORE DISTRIBUTION
# How well does the model separate high vs low risk?
# You want the claim=1 distribution shifted right
# ============================================================

ggplot(val_preds, aes(x = xgb_pred, fill = factor(actual))) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c("#2C7BB6", "#D7191C"),
                    labels = c("No Claim", "Claim")) +
  labs(
    title    = "Predicted Probability Distribution by Actual Outcome",
    subtitle = "XGBoost — claim group should skew higher",
    x        = "Predicted Claim Probability",
    y        = "Density",
    fill     = "Actual Outcome"
  ) +
  theme_minimal()

ggsave("plots/11_score_distribution.png", width = 7, height = 5)


# ============================================================
# 4. EXPERIENCE ANALYSIS BY SUBGROUP
# This is the actuarial diagnostic — check whether the model
# over or under predicts for specific subgroups
# Actuaries call this "actual vs expected" (A/E analysis)
# ============================================================

# --- By ps_car_11_cat (if it exists) or use ps_ind_02_cat ---
ae_by_cat <- val_full %>%
  mutate(ps_ind_02_cat = as.factor(ps_ind_02_cat)) %>%
  filter(!is.na(ps_ind_02_cat)) %>%
  group_by(ps_ind_02_cat) %>%
  summarise(
    n              = n(),
    actual_rate    = mean(target_num),
    predicted_rate = mean(xgb_pred),
    ae_ratio       = actual_rate / predicted_rate,
    .groups = "drop"
  ) %>%
  filter(n > 500)   # only subgroups with enough exposure to be credible

print(ae_by_cat)

ggplot(ae_by_cat, aes(x = ps_ind_02_cat, y = ae_ratio)) +
  geom_col(aes(fill = ae_ratio > 1.05 | ae_ratio < 0.95)) +
  geom_hline(yintercept = 1, linetype = "dashed",
             color = "black", linewidth = 1) +
  scale_fill_manual(values = c("#2C7BB6", "#D7191C"),
                    labels = c("Within 5% — adequate",
                               "Outside 5% — model gap")) +
  labs(
    title    = "Actual vs Expected Ratio by Category (ps_ind_02_cat)",
    subtitle = "A/E ratio > 1 means model is under-predicting risk",
    x        = "Category",
    y        = "A/E Ratio (Actual / Predicted)",
    fill     = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/12_ae_analysis.png", width = 7, height = 5)


# ============================================================
# 5. CALIBRATION PLOT
# Does the model's predicted probability match reality?
# Group predictions into buckets and compare to actual rate
# ============================================================

calibration_data <- val_preds %>%
  mutate(pred_bucket = ntile(xgb_pred, 20)) %>%
  group_by(pred_bucket) %>%
  summarise(
    mean_predicted = mean(xgb_pred),
    mean_actual    = mean(actual),
    n              = n(),
    .groups = "drop"
  )

ggplot(calibration_data, aes(x = mean_predicted, y = mean_actual)) +
  geom_point(aes(size = n), color = "#2C7BB6", alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              color = "red") +
  labs(
    title    = "Calibration Plot — XGBoost",
    subtitle = "Points on red line = perfectly calibrated predictions",
    x        = "Mean Predicted Probability",
    y        = "Mean Actual Claim Rate",
    size     = "n"
  ) +
  theme_minimal()

ggsave("plots/13_calibration.png", width = 7, height = 5)


# ============================================================
# 6. DOUBLE LIFT CHART
# Compare XGBoost vs Logistic GLM head to head
# Shows where the models agree and disagree on risk ranking
# ============================================================

double_lift <- val_preds %>%
  mutate(
    xgb_decile      = ntile(desc(xgb_pred), 10),
    logistic_decile = ntile(desc(logistic_pred), 10)
  ) %>%
  group_by(xgb_decile) %>%
  summarise(
    xgb_claim_rate      = mean(actual[xgb_decile == xgb_decile]),
    logistic_claim_rate = mean(actual[logistic_decile == xgb_decile]),
    .groups = "drop"
  )

# Cleaner approach
double_lift <- bind_rows(
  val_preds %>%
    mutate(decile = ntile(desc(xgb_pred), 10)) %>%
    group_by(decile) %>%
    summarise(claim_rate = mean(actual), model = "XGBoost", .groups = "drop"),
  val_preds %>%
    mutate(decile = ntile(desc(logistic_pred), 10)) %>%
    group_by(decile) %>%
    summarise(claim_rate = mean(actual), model = "Logistic GLM", .groups = "drop")
)

ggplot(double_lift, aes(x = decile, y = claim_rate, color = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_color_manual(values = c("#D7191C", "#2C7BB6")) +
  scale_x_continuous(breaks = 1:10) +
  labs(
    title    = "Double Lift Chart — XGBoost vs Logistic GLM",
    subtitle = "Decile 1 = highest predicted risk",
    x        = "Risk Decile",
    y        = "Actual Claim Rate",
    color    = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/14_double_lift.png", width = 8, height = 5)

cat("Diagnostics complete. All plots saved to /plots/\n")