# Load Libraries (once)
library(VIM)
library(corrplot)
library(car)
library(psych)
library(factoextra)
library(MASS)
library(dplyr)
library(tidyverse)
library(caret)
library(plotly)
library(ggplot2)
library(GGally)
library(CCA)
library(pROC)
library(glmnet)
library(leaps)  # For regsubsets
library(gridExtra)

# Import Dataset
merged_file <- read.csv("merged_file.csv")
summary(merged_file)
df <- data.frame(merged_file)

# Data Preprocessing
# Handle missing data with kNN imputation
df_imputed <- kNN(df, k = 2, numFun = median, imp_var = FALSE)
summary(df_imputed)

# Split dataset into offensive, defensive, and general variables
df_offensive <- df_imputed[, c("R", "AB", "H", "X2B", "X3B", "HR", "BB", "SO", "SB")]
df_defensive <- df_imputed[, c("RA", "ER", "ERA", "CG", "SHO", "SV", "IPouts", "HA", "HRA", "BBA", "SOA", "E", "DP", "FP")]
df_general <- df_imputed[, c("yearID", "G", "Ghome", "W", "L", "Rank", "DivWin", "WCWin", "LgWin", "WSWin", "BPF", "PPF", "attendance")]

# Exploratory Data Analysis
# Correlation plots
png("corrplot_defensive.png")
corrplot(cor(df_defensive), order = "AOE")  # Figure 3
dev.off()

png("corrplot_offensive.png")
corrplot(cor(df_offensive), order = "AOE")  # Figure 4
dev.off()

# GGPairs for visualization
p <- ggpairs(df_defensive)  # Figure 1
ggsave("ggpairs_defensive.png", p)

p <- ggpairs(df_offensive)  # Figure 2
ggsave("ggpairs_offensive.png", p)

# Suitability for factor analysis
KMO(df_offensive)
KMO(df_defensive)

# Regularized Regression Approach (Defensive Variables)
Defensive_vars <- df_imputed[, c("teamID", "W", "RA", "ER", "ERA", "CG", "SHO", "SV", "IPouts", "HA", "HRA", "BBA", "SOA", "E", "DP", "FP")]
y <- Defensive_vars$W
X <- model.matrix(W ~ . - teamID, Defensive_vars)[, -1]  # Exclude teamID
X <- scale(X)

# Training and testing sets
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Ridge Regression
lambda_seq <- 10^seq(2, -2, by = -0.1)
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, lambda = lambda_seq)
optimal_lambda_ridge <- ridge_cv$lambda.min
ridge_pred <- predict(ridge_cv, s = optimal_lambda_ridge, newx = X_test)
ridge_mse <- mean((ridge_pred - y_test)^2)
print(ridge_cv)
png("ridge_cv_defensive.png")
plot(ridge_cv)
dev.off()

# Lasso Regression
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, lambda = lambda_seq)
optimal_lambda_lasso <- lasso_cv$lambda.min
lasso_pred <- predict(lasso_cv, s = optimal_lambda_lasso, newx = X_test)
lasso_mse <- mean((lasso_pred - y_test)^2)
print(lasso_cv)
png("lasso_cv_defensive.png")
plot(lasso_cv)
dev.off()

# Elastic Net Regression
elastic_net_cv <- cv.glmnet(X_train, y_train, alpha = 0.5, lambda = lambda_seq)
optimal_lambda_elastic <- elastic_net_cv$lambda.min
elastic_pred <- predict(elastic_net_cv, s = optimal_lambda_elastic, newx = X_test)
elastic_mse <- mean((elastic_pred - y_test)^2)
cat("Elastic Net MSE: ", elastic_mse, "\n")
cat("Optimal Lambda for Elastic Net: ", optimal_lambda_elastic, "\n")
print(coef(elastic_net_cv, s = optimal_lambda_elastic))
png("elastic_net_cv_defensive.png")
plot(elastic_net_cv)
dev.off()

# All Subsets Regression
subset_model <- regsubsets(W ~ . - teamID, data = Defensive_vars, nvmax = 10)
subset_summary <- summary(subset_model)
png("subset_adjr2_defensive.png")
plot(subset_model, scale = "adjr2")  # Adjusted R-squared
dev.off()
png("subset_bic_defensive.png")
plot(subset_model, scale = "bic")    # BIC
dev.off()

# Best subset model
best_subset <- which.max(subset_summary$adjr2)
cat("Best subset of variables (Adjusted R-squared):\n")
print(names(Defensive_vars)[subset_summary$which[best_subset, ]][-1])  # Exclude intercept

# Fit linear model with best subset
model_all_subset <- lm(W ~ RA + CG + SV + IPouts, data = Defensive_vars)
summary(model_all_subset)
vif(model_all_subset)

# Stepwise Regression
full_model <- lm(W ~ . - teamID, data = Defensive_vars)
forward_model <- step(lm(W ~ 1, data = Defensive_vars), scope = list(lower = ~1, upper = full_model), direction = "forward", trace = FALSE)
summary(forward_model)
backward_model <- step(full_model, direction = "backward", trace = FALSE)
summary(backward_model)

# Regression on Offensive Variables
Offensive_vars <- df_imputed[, c("W","R", "AB", "H", "X2B", "X3B", "HR", "BB", "SO", "SB")]
y <- Offensive_vars$W
X <- model.matrix(W ~ ., Offensive_vars)[, -1]
X <- scale(X)

# Training and testing sets
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Ridge Regression
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, lambda = lambda_seq)
optimal_lambda_ridge <- ridge_cv$lambda.min
ridge_pred <- predict(ridge_cv, s = optimal_lambda_ridge, newx = X_test)
ridge_mse <- mean((ridge_pred - y_test)^2)
print(ridge_cv)
png("ridge_cv_offensive.png")
plot(ridge_cv)
dev.off()

# Lasso Regression
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, lambda = lambda_seq)
optimal_lambda_lasso <- lasso_cv$lambda.min
lasso_pred <- predict(lasso_cv, s = optimal_lambda_lasso, newx = X_test)
lasso_mse <- mean((lasso_pred - y_test)^2)
print(lasso_cv)
png("lasso_cv_offensive.png")
plot(lasso_cv)
dev.off()

# Elastic Net Regression
elastic_net_cv <- cv.glmnet(X_train, y_train, alpha = 0.5, lambda = lambda_seq)
optimal_lambda_elastic <- elastic_net_cv$lambda.min
elastic_pred <- predict(elastic_net_cv, s = optimal_lambda_elastic, newx = X_test)
elastic_mse <- mean((elastic_pred - y_test)^2)
cat("Elastic Net MSE: ", elastic_mse, "\n")
print(coef(elastic_net_cv, s = optimal_lambda_elastic))
png("elastic_net_cv_offensive.png")
plot(elastic_net_cv)
dev.off()

# Factor Analysis and PCA
# Scale data
df_offensive_s <- scale(df_offensive)
df_defensive_s <- scale(df_defensive)

# PCA
pca_off <- prcomp(df_offensive_s, scale. = TRUE)
summary(pca_off)
png("pca_offensive.png")
plot(pca_off)
abline(h = 1, col = "red")
dev.off()

pca_def <- prcomp(df_defensive_s, scale. = TRUE)
summary(pca_def)
png("pca_defensive.png")
plot(pca_def)
abline(h = 1, col = "red")
dev.off()

# Principal Factor Analysis with Varimax Rotation
pfa_off <- principal(df_offensive_s, nfactors = 2, rotate = "varimax")
print(pfa_off$loadings, cutoff = 0.4)
pfa_def <- principal(df_defensive_s, nfactors = 3, rotate = "varimax")
print(pfa_def$loadings, cutoff = 0.4)

# Create datasets with factor scores
Off_data <- as.data.frame(pfa_off$scores)
names(Off_data) <- c("Batting_Efficiency", "Aggressive_VS_Pace_Plays")
Def_data <- as.data.frame(pfa_def$scores)
names(Def_data) <- c("Run_Control", "Pitching_Efficiency", "Shutout_Plays")

# Add metadata
Off_data <- cbind(Off_data, teamID = merged_file$teamID, Rank = merged_file$Rank, Win = merged_file$W, Year = merged_file$yearID)
Def_data <- cbind(Def_data, teamID = merged_file$teamID, Rank = merged_file$Rank, Win = merged_file$W, Year = merged_file$yearID)

# Define Eras
df_imputed$Era <- cut(df_imputed$yearID, breaks = c(1871, 1920, 1947, 1990, 2023), 
                      labels = c("Deadball Era", "Segregation Era", "Post Segregation Era", "Modern Era"), 
                      include.lowest = TRUE)
Off_data$Era <- df_imputed$Era
Def_data$Era <- df_imputed$Era

# Final Analysis
# Merge offensive and defensive data
merged_d <- Off_data %>%
  left_join(select(Def_data, teamID, Year, Run_Control, Pitching_Efficiency, Shutout_Plays, Era), 
            by = c("teamID", "Year", "Era"))

# Boxplots by Era
p1 <- ggplot(merged_d, aes(x = Era, y = Batting_Efficiency, fill = Era)) +
  geom_boxplot() + labs(title = "Batting Efficiency Across Eras") + theme_minimal() +
  scale_fill_manual(values = c("blue", "green", "orange", "red")) + theme(legend.position = "none")
ggsave("boxplot_batting_efficiency.png", p1)

p2 <- ggplot(merged_d, aes(x = Era, y = Pitching_Efficiency, fill = Era)) +
  geom_boxplot() + labs(title = "Pitching Efficiency Across Eras") + theme_minimal() +
  scale_fill_manual(values = c("blue", "green", "orange", "red")) + theme(legend.position = "none")
ggsave("boxplot_pitching_efficiency.png", p2)

# Combine boxplots into one file
grid_plot <- grid.arrange(p1, p2, ncol = 2)
ggsave("boxplots_combined.png", grid_plot)

# Victory Drivers Plot
highlight_teams <- merged_d %>%
  filter((teamID == "CHN" & Year == 1906) | (teamID == "NYA" & Year == 1950) | (teamID == "LAN" & Year == 2022)) %>%
  mutate(victory_driver = case_when(
    teamID == "CHN" & Year == 1906 ~ "Pitching",
    teamID == "NYA" & Year == 1950 ~ "Offense",
    teamID == "LAN" & Year == 2022 ~ "Synergy",
    TRUE ~ NA_character_
  ))

p_victory <- ggplot(merged_d, aes(x = Batting_Efficiency, y = Shutout_Plays, color = Era)) +
  geom_point(alpha = 0.3) +
  geom_point(data = highlight_teams, size = 6, shape = 21, fill = "yellow", color = "black") +
  geom_text(data = highlight_teams, aes(label = paste(teamID, Year)), vjust = -1.5, color = "black", fontface = "bold") +
  geom_text(data = highlight_teams, aes(label = victory_driver), vjust = -0.5, color = "darkgray") +
  labs(title = "Victory Drivers", x = "Batting Efficiency", y = "Shutout Plays") +
  scale_color_manual(values = c("blue", "green", "orange", "red")) +
  theme_minimal()
ggsave("victory_drivers.png", p_victory)

# Canonical Correlation Analysis (CCA)
# Define variables for CCA
offensive_vars <- c("R", "AB", "H", "X2B", "X3B", "HR", "BB", "SO")
defensive_vars <- c("RA", "ER", "ERA", "CG", "SHO", "SV", "IPouts", "HA", "HRA", "BBA", "SOA", "E", "DP", "FP")

# Function to perform CCA by era
cca_tradeoffs_by_era <- function(data, era) {
  df_era <- data %>%
    filter(Era == era) %>%
    select(Era, DivWin, all_of(offensive_vars), all_of(defensive_vars)) %>%
    na.omit()
  
  set.seed(42)
  train_idx <- createDataPartition(df_era$DivWin, p = 0.7, list = FALSE)
  train_data <- df_era[train_idx, ]
  test_data <- df_era[-train_idx, ]
  
  X1 <- as.matrix(train_data[, offensive_vars])
  X2 <- as.matrix(train_data[, defensive_vars])
  cca_result <- cancor(X1, X2)
  
  U1 <- X1 %*% cca_result$xcoef[, 1]
  V1 <- X2 %*% cca_result$ycoef[, 1]
  df_cca_train <- data.frame(DivWin = train_data$DivWin, U1 = U1, V1 = V1)
  
  X1_test <- as.matrix(test_data[, offensive_vars])
  X2_test <- as.matrix(test_data[, defensive_vars])
  U1_test <- X1_test %*% cca_result$xcoef[, 1]
  V1_test <- X2_test %*% cca_result$ycoef[, 1]
  df_cca_test <- data.frame(DivWin = test_data$DivWin, U1 = U1_test, V1 = V1_test)
  
  cat("\nEra:", era, "\nCanonical Correlation:", cca_result$cor[1], "\n")
  cat("Offensive Loadings (U1):\n")
  print(cca_result$xcoef[, 1])
  cat("Defensive Loadings (V1):\n")
  print(cca_result$ycoef[, 1])
  
  return(list(cca_result = cca_result, train_data = df_cca_train, test_data = df_cca_test))
}

# Run CCA for all eras
eras <- c("Deadball Era", "Segregation Era", "Post Segregation Era", "Modern Era")
cca_tradeoffs_results <- lapply(eras, function(e) cca_tradeoffs_by_era(df_imputed, e))
names(cca_tradeoffs_results) <- eras

# Goodness of Fit for CCA
cca_goodness_of_fit <- function(cca_results, era) {
  train_data <- cca_results[[era]]$train_data
  test_data <- cca_results[[era]]$test_data
  cca_result <- cca_results[[era]]$cca_result
  
  can_cor <- cca_result$cor[1]
  
  log_model <- glm(DivWin ~ U1 + V1, data = train_data, family = "binomial")
  pred_prob <- predict(log_model, test_data, type = "response")
  pred_class <- ifelse(pred_prob > 0.5, "Y", "N")
  confusion <- table(Predicted = pred_class, Actual = test_data$DivWin)
  accuracy <- sum(diag(confusion)) / sum(confusion)
  
  roc_obj <- roc(as.numeric(test_data$DivWin == "Y"), pred_prob)
  auc_value <- auc(roc_obj)
  
  cat("\nEra:", era, "\n")
  cat("Canonical Correlation (U1-V1):", can_cor, "\n")
  cat("Logistic Regression Accuracy:", accuracy, "\n")
  cat("AUC:", auc_value, "\n")
  
  return(list(accuracy = accuracy, auc = auc_value))
}

# Run goodness of fit for all eras
fit_results <- lapply(eras, function(e) cca_goodness_of_fit(cca_tradeoffs_results, e))
names(fit_results) <- eras