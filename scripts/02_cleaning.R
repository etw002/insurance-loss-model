# ============================================================
# 02_cleaning.R
# Porto Seguro - Data Cleaning & Feature Engineering
# ============================================================

library(tidyverse)

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

train <- read_csv("data/train.csv")

# ============================================================
# 2. REPLACE -1 WITH NA
# Porto Seguro uses -1 as a missing value placeholder
# We need to convert these to proper NAs before doing anything else
# ============================================================

train_clean <- train %>%
  mutate(across(everything(), ~ ifelse(. == -1, NA, .)))

# Confirm -1s are gone
sum(train_clean == -1, na.rm = TRUE)  # should return 0


# ============================================================
# 3. DROP HIGH-MISSINGNESS VARIABLES
# ps_car_03_cat (~70% missing) and ps_car_05_cat (~45% missing)
# Too much data missing to impute reliably — dropping both
# This is a deliberate modeling decision, note it in your README
# ============================================================

train_clean <- train_clean %>%
  select(-ps_car_03_cat, -ps_car_05_cat)

cat("Columns remaining:", ncol(train_clean), "\n")


# ============================================================
# 4. HANDLE REMAINING MISSING VALUES
# For variables with moderate missingness, we impute:
#   - Categorical/binary: impute with mode (most common value)
#   - Continuous: impute with median (more robust than mean)
# ============================================================

# --- Categorical and binary: mode imputation ---
# Build a helper function for mode
get_mode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

cat_bin_vars <- names(train_clean)[
  str_detect(names(train_clean), "_cat|_bin")
]

train_clean <- train_clean %>%
  mutate(across(
    all_of(cat_bin_vars),
    ~ ifelse(is.na(.), get_mode(.), .)
  ))

# --- Continuous: median imputation ---
reg_vars <- names(train_clean)[
  str_detect(names(train_clean), "_reg")
]

train_clean <- train_clean %>%
  mutate(across(
    all_of(reg_vars),
    ~ ifelse(is.na(.), median(., na.rm = TRUE), .)
  ))

# cleaning unsuffixed vars
unsuffixed_vars <- c("ps_car_11", "ps_car_12", "ps_car_14")

train_clean <- train_clean %>%
  mutate(across(
    all_of(unsuffixed_vars),
    ~ ifelse(is.na(.), median(., na.rm = TRUE), .)
  ))

# Confirm no NAs remain
cat("Remaining NAs:", sum(is.na(train_clean)), "\n")  # should be 0 or close

# Returns a character vector of column names
names(train_clean)[colSums(is.na(train_clean)) > 0]

# ============================================================
# 5. CONVERT CATEGORICALS TO FACTORS
# GLMs and tree models both need to know which variables
# are categorical vs continuous — factors tell R that
# ============================================================

train_clean <- train_clean %>%
  mutate(across(all_of(cat_bin_vars), as.factor))

# Also make target a factor for classification models later
# We'll keep a numeric version too for the Poisson GLM
train_clean <- train_clean %>%
  mutate(
    target_num = target,          # keep numeric for Poisson GLM
    target     = as.factor(target)
  )


# ============================================================
# 6. FEATURE ENGINEERING
# Create 2 new variables that might have predictive signal
# Documenting your reasoning is important — actuaries justify
# every variable they include in a rate filing
# ============================================================

# Feature 1: Total missing flag count per row
# Policyholders with lots of missing info might be higher risk
# (they may have withheld information)
raw_train <- read_csv("data/train.csv")  # reload raw to count original -1s

missing_flags <- raw_train %>%
  mutate(missing_count = rowSums(. == -1, na.rm = TRUE)) %>%
  select(id, missing_count)

train_clean <- train_clean %>%
  left_join(missing_flags, by = "id")


# Feature 2: Interaction between ps_car_13 and ps_reg_03
# ps_car_13 is strongly correlated with car value
# ps_reg_03 is related to region risk
# Their interaction captures "expensive car in high-risk region"
train_clean <- train_clean %>%
  mutate(
    car_region_interaction = ps_car_13 * ps_reg_03
  )

# Quick sanity check on engineered features
train_clean %>%
  group_by(target) %>%
  summarise(
    avg_missing_count        = mean(missing_count, na.rm = TRUE),
    avg_car_region           = mean(car_region_interaction, na.rm = TRUE)
  )
# If the engineered features have different means between claim/no-claim
# groups, they have predictive signal -- good sign


# ============================================================
# 7. TRAIN / VALIDATION SPLIT
# 80% train, 20% validation
# Set a seed so your results are reproducible
# ============================================================

set.seed(42)

n <- nrow(train_clean)
train_idx <- sample(1:n, size = floor(0.8 * n))

train_final <- train_clean[train_idx, ]
val_final   <- train_clean[-train_idx, ]

cat("Training rows:", nrow(train_final), "\n")
cat("Validation rows:", nrow(val_final), "\n")

# Check that claim rate is roughly preserved in both splits
cat("Train claim rate:", mean(train_final$target_num), "\n")
cat("Val claim rate:  ", mean(val_final$target_num), "\n")
# These should both be close to 0.0364


# ============================================================
# 8. SAVE CLEANED DATA
# ============================================================

write_csv(train_clean, "data/train_final.csv")

cat("Cleaned data saved.\n")