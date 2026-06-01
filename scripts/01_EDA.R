# ============================================================
# 01_eda.R
# Porto Seguro - Exploratory Data Analysis
# ============================================================

library(tidyverse)
library(ggplot2)
library(corrplot)
library(patchwork)

# ============================================================
# 1. Load data
# ============================================================

train <- read_csv("data/train.csv")

# First look
glimpse(train)
dim(train)
head(train)
summary(train)


# ============================================================
# 2. TARGET VARIABLE - Claim Frequency
# ============================================================

# Overall claim rate
train %>%
  count(target) %>%
  mutate(pct = n / sum(n) * 100)

# Visualize
ggplot(train, aes(x = factor(target))) +
  geom_bar(fill = c("#2C7BB6", "#D7191C")) +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  labs(
    title = "Target Variable Distribution",
    subtitle = "0 = No Claim, 1 = Claim Filed",
    x = "Claim Filed",
    y = "Count"
  ) +
  theme_minimal()

ggsave("plots/01_target_distribution.png", width = 6, height = 4)

# You'll notice heavy class imbalance — roughly 96% no claim, 4% claim
# Write this down, it matters when we model


# ============================================================
# 3. MISSING VALUES
# ============================================================

# Porto Seguro uses -1 as a missing value placeholder
# Count how many -1s exist per column

missing_counts <- train %>%
  summarise(across(everything(), ~ sum(. == -1, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
  filter(missing_count > 0) %>%
  mutate(missing_pct = missing_count / nrow(train) * 100) %>%
  arrange(desc(missing_pct))

print(missing_counts)

# Visualize missingness
ggplot(missing_counts, aes(x = reorder(variable, missing_pct), y = missing_pct)) +
  geom_col(fill = "#D7191C") +
  coord_flip() +
  labs(
    title = "Variables with Missing Values (-1 encoded)",
    x = "Variable",
    y = "% Missing"
  ) +
  theme_minimal()

ggsave("plots/02_missing_values.png", width = 7, height = 5)


# ============================================================
# 4. UNDERSTAND VARIABLE TYPES
# Porto Seguro uses suffixes to tell you what each variable is:
#   _bin = binary (0/1)
#   _cat = categorical
#   _reg = continuous/regression
#   no suffix = ordinal or continuous
# ============================================================

bin_vars  <- names(train)[str_detect(names(train), "_bin")]
cat_vars  <- names(train)[str_detect(names(train), "_cat")]
reg_vars  <- names(train)[str_detect(names(train), "_reg")]

cat("Binary vars:", length(bin_vars), "\n")
cat("Categorical vars:", length(cat_vars), "\n")
cat("Continuous vars:", length(reg_vars), "\n")


# ============================================================
# 5. CLAIM RATE BY BINARY VARIABLES
# ============================================================

# For each binary variable, calculate claim rate when 0 vs 1
bin_claim_rates <- train %>%
  select(target, all_of(bin_vars)) %>%
  pivot_longer(-target, names_to = "variable", values_to = "value") %>%
  filter(value != -1) %>%           # remove missing
  group_by(variable, value) %>%
  summarise(
    claim_rate = mean(target),
    n = n(),
    .groups = "drop"
  )

ggplot(bin_claim_rates, aes(x = factor(value), y = claim_rate, fill = factor(value))) +
  geom_col() +
  facet_wrap(~variable, scales = "free_y") +
  labs(
    title = "Claim Rate by Binary Variables",
    x = "Value",
    y = "Claim Rate"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("plots/03_claim_rate_binary.png", width = 12, height = 8)


# ============================================================
# 6. CLAIM RATE BY CATEGORICAL VARIABLES
# ============================================================

cat_claim_rates <- train %>%
  select(target, all_of(cat_vars)) %>%
  pivot_longer(-target, names_to = "variable", values_to = "value") %>%
  filter(value != -1) %>%
  group_by(variable, value) %>%
  summarise(
    claim_rate = mean(target),
    n = n(),
    .groups = "drop"
  )

ggplot(cat_claim_rates, aes(x = factor(value), y = claim_rate)) +
  geom_col(fill = "#2C7BB6") +
  geom_hline(yintercept = mean(train$target), linetype = "dashed", color = "red") +
  facet_wrap(~variable, scales = "free_x") +
  labs(
    title = "Claim Rate by Categorical Variables",
    subtitle = "Red dashed line = overall average claim rate",
    x = "Category",
    y = "Claim Rate"
  ) +
  theme_minimal()

ggsave("plots/04_claim_rate_categorical.png", width = 14, height = 10)


# ============================================================
# 7. DISTRIBUTION OF CONTINUOUS VARIABLES
# ============================================================

reg_long <- train %>%
  select(target, all_of(reg_vars)) %>%
  mutate(across(all_of(reg_vars), ~ ifelse(. == -1, NA, .))) %>%
  pivot_longer(-target, names_to = "variable", values_to = "value") %>%
  drop_na()

ggplot(reg_long, aes(x = value, fill = factor(target))) +
  geom_density(alpha = 0.5) +
  facet_wrap(~variable, scales = "free") +
  scale_fill_manual(values = c("#2C7BB6", "#D7191C"),
                    labels = c("No Claim", "Claim")) +
  labs(
    title = "Distribution of Continuous Variables by Claim Status",
    fill = "Claim Status",
    x = "Value",
    y = "Density"
  ) +
  theme_minimal()

ggsave("plots/05_continuous_distributions.png", width = 14, height = 10)


# ============================================================
# 8. CORRELATION HEATMAP (continuous vars only)
# ============================================================

cor_data <- train %>%
  select(target, all_of(reg_vars)) %>%
  mutate(across(everything(), ~ ifelse(. == -1, NA, .))) %>%
  drop_na()

cor_matrix <- cor(cor_data)

png("plots/06_correlation_heatmap.png", width = 900, height = 800)
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 0.7,
         tl.col = "black",
         title = "Correlation Matrix - Continuous Variables",
         mar = c(0,0,2,0))
dev.off()


# ============================================================
# 9. SUMMARY NOTES — write these down for your README later
# ============================================================

# Things to note after running this:
# - What is the exact claim rate? (should be ~3.6%)
# - Which binary/categorical variables show the biggest spread in claim rate?
# - Which continuous variables look most different between claim vs no claim?
# - Which variables have the most missing data? (ps_car_03_cat and ps_car_05_cat are bad)
# - Any variables so skewed or missing they should be dropped?