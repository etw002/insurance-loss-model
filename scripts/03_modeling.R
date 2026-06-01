# ============================================================
# 03_modeling.R
# Porto Seguro - Modeling
# Three models: Logistic GLM, Poisson GLM, XGBoost
# Evaluation: AUC, Gini Coefficient, Lift Curve
# ============================================================

library(tidyverse)
library(ggplot2)
library(patchwork)
library(xgboost)
library(pROC)

# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

train_final <- read_csv("data/train_final.csv")
test_final   <- read_csv("data/test.csv")

# Re-apply factor conversion after reading from CSV
cat_bin_vars <- names(train_final)[str_detect(names(train_final), "_cat|_bin")]

train_final <- train_final %>%
  mutate(across(all_of(cat_bin_vars), as.factor),
         target = as.factor(target))

val_final <- val_final %>%
  mutate(across(all_of(cat_bin_vars), as.factor),
         target = as.factor(target))

# Drop id column — not a predictor
train_model <- train_final %>% select(-id)
val_model   <- val_final   %>% select(-id)


# ============================================================
# 2. MODEL 1 — LOGISTIC GLM (Baseline)
# Standard binary classification
# This is your actuarial baseline — GLMs are the industry
# standard for P&C pricing models
# ============================================================

cat("Fitting Logistic GLM...\n")

logistic_glm <- glm(
  target ~ . - target_num,
  data   = train_model,
  family = binomial(link = "logit")
)

summary(logistic_glm)

# Predict probabilities on validation set
logistic_preds <- predict(logistic_glm, newdata = val_model, type = "response")

# AUC
logistic_roc  <- roc(as.numeric(val_model$target) - 1, logistic_preds)
logistic_auc  <- auc(logistic_roc)
logistic_gini <- 2 * logistic_auc - 1

cat("Logistic GLM  | AUC:", round(logistic_auc, 4),
    "| Gini:", round(logistic_gini, 4), "\n")


# ============================================================
# 3. MODEL 2 — POISSON GLM (Frequency Model)
# Reframes the problem as claim frequency rather than
# binary classification — this is closer to how actuaries
# actually build pricing models
# ============================================================

cat("Fitting Poisson GLM...\n")

# Use numeric target for Poisson
train_poisson <- train_model %>% mutate(target = target_num)
val_poisson   <- val_model   %>% mutate(target = as.numeric(as.character(target)))

poisson_glm <- glm(
  target ~ . - target_num,
  data   = train_poisson,
  family = poisson(link = "log")
)

summary(poisson_glm)

poisson_preds <- predict(poisson_glm, newdata = val_poisson, type = "response")

poisson_roc  <- roc(val_poisson$target, poisson_preds)
poisson_auc  <- auc(poisson_roc)
poisson_gini <- 2 * poisson_auc - 1

cat("Poisson GLM   | AUC:", round(poisson_auc, 4),
    "| Gini:", round(poisson_gini, 4), "\n")


# ============================================================
# 4. MODEL 3 — XGBOOST
# Gradient boosted trees — modern ML comparison
# XGBoost often outperforms GLMs on raw AUC but is less
# interpretable. Important to know both and the tradeoffs.
# ============================================================

cat("Fitting XGBoost...\n")

# XGBoost needs a numeric matrix — convert factors to integers
prep_xgb <- function(df) {
  df %>%
    select(-target, -target_num) %>%
    mutate(across(where(is.factor), as.integer)) %>%
    as.matrix()
}

train_xgb_mat <- prep_xgb(train_model)
val_xgb_mat   <- prep_xgb(val_model)

train_xgb_label <- as.numeric(as.character(train_model$target))
val_xgb_label   <- as.numeric(as.character(val_model$target))

dtrain <- xgb.DMatrix(data = train_xgb_mat, label = train_xgb_label)
dval   <- xgb.DMatrix(data = val_xgb_mat,   label = val_xgb_label)

xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,       # learning rate — lower = more robust
  max_depth        = 5,          # tree depth
  subsample        = 0.8,        # row sampling per tree
  colsample_bytree = 0.8,        # column sampling per tree
  min_child_weight = 5           # prevents overfitting on rare events
)

set.seed(42)
xgb_model <- xgb.train(
  params  = xgb_params,
  data    = dtrain,
  nrounds = 300,
  watchlist     = list(val = dval),
  early_stopping_rounds = 20,   # stops if val AUC doesn't improve
  verbose = 1
)

xgb_preds <- predict(xgb_model, dval)

xgb_roc  <- roc(val_xgb_label, xgb_preds)
xgb_auc  <- auc(xgb_roc)
xgb_gini <- 2 * xgb_auc - 1

cat("XGBoost       | AUC:", round(xgb_auc, 4),
    "| Gini:", round(xgb_gini, 4), "\n")


# ============================================================
# 5. MODEL COMPARISON TABLE
# ============================================================

results <- tibble(
  Model = c("Logistic GLM", "Poisson GLM", "XGBoost"),
  AUC   = c(logistic_auc, poisson_auc, xgb_auc),
  Gini  = c(logistic_gini, poisson_gini, xgb_gini)
) %>%
  mutate(across(c(AUC, Gini), ~ round(., 4)))

print(results)

ggplot(results, aes(x = reorder(Model, Gini), y = Gini, fill = Model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(Gini, 3)), hjust = -0.2, size = 4) +
  coord_flip() +
  scale_fill_manual(values = c("#2C7BB6", "#ABD9E9", "#D7191C")) +
  labs(
    title = "Model Comparison — Gini Coefficient",
    subtitle = "Higher is better | Gini = 2×AUC − 1",
    x = NULL,
    y = "Gini Coefficient"
  ) +
  theme_minimal() +
  theme(legend.position = "none") +
  ylim(0, max(results$Gini) * 1.2)

ggsave("plots/07_model_comparison.png", width = 7, height = 4)


# ============================================================
# 6. LIFT CURVE — Most important actuarial diagnostic
# Sort policyholders by predicted risk, split into deciles,
# compare actual claim rate in each decile vs average
# A good model should show top decile at 2-3x average claim rate
# ============================================================

lift_data <- tibble(
  actual    = val_xgb_label,
  predicted = xgb_preds
) %>%
  arrange(desc(predicted)) %>%
  mutate(
    decile         = ntile(desc(predicted), 10),
    overall_rate   = mean(actual)
  ) %>%
  group_by(decile) %>%
  summarise(
    actual_claim_rate  = mean(actual),
    overall_rate       = first(overall_rate),
    lift               = actual_claim_rate / overall_rate,
    n                  = n(),
    .groups = "drop"
  )

print(lift_data)

ggplot(lift_data, aes(x = decile, y = lift)) +
  geom_col(fill = "#2C7BB6") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 1) +
  geom_text(aes(label = round(lift, 2)), vjust = -0.4, size = 3.5) +
  scale_x_continuous(breaks = 1:10) +
  labs(
    title    = "Lift Curve by Risk Decile — XGBoost",
    subtitle = "Decile 1 = highest predicted risk | Red line = no-model baseline",
    x        = "Risk Decile (1 = Highest Risk)",
    y        = "Lift (Actual Rate / Average Rate)"
  ) +
  theme_minimal()

ggsave("plots/08_lift_curve.png", width = 8, height = 5)


# ============================================================
# 7. FEATURE IMPORTANCE — XGBoost
# ============================================================

importance_matrix <- xgb.importance(
  feature_names = colnames(train_xgb_mat),
  model         = xgb_model
)

# Top 20 features
top20 <- importance_matrix %>%
  as_tibble() %>%
  slice_head(n = 20)

ggplot(top20, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#2C7BB6") +
  coord_flip() +
  labs(
    title    = "XGBoost Feature Importance (Top 20)",
    subtitle = "Gain = average improvement in loss brought by a feature",
    x        = NULL,
    y        = "Gain"
  ) +
  theme_minimal()

ggsave("plots/09_feature_importance.png", width = 8, height = 6)


# ============================================================
# 8. SAVE PREDICTIONS FOR DIAGNOSTICS SCRIPT
# ============================================================

val_predictions <- tibble(
  actual          = val_xgb_label,
  xgb_pred        = xgb_preds,
  logistic_pred   = logistic_preds,
  poisson_pred    = poisson_preds
)

write_csv(val_predictions, "data/val_predictions.csv")
write_csv(results,         "data/model_results.csv")

cat("Modeling complete. Predictions saved.\n")