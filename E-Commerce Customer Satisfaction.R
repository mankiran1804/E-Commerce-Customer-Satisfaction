#Load necessary libraries
library(ggplot2)
library(dplyr)
library(MASS)  # for LDA
library(caret)  # for cross-validation
library(leaps)
library(glmnet)  # for potential regularization needs
library(car)     # for additional diagnostic plots
library(MASS)  # For stepwise regression
#install.packages("brglm")
library(brglm)  # for bias Sreduction in logistic regression
library(boot)   # for bootstrap
library(class)
library(GGally)      # For scatterplot matrix
library(corrplot)    # For correlation matrix plot

# Set path for the dataset
data <- read.csv("C:/Users/dipsyrd9876/Downloads/ECommerceCustomerBehavior.csv")

####################
# EDA
####################
# Univariate Analysis
# Theme setup for light gray background
custom_theme <- theme_minimal() +
  theme(panel.background = element_rect(fill = "lightgray"),
        panel.border = element_rect(colour = "black", fill = NA, size = 0.5))

# Univariate Analysis
p_age <- ggplot(data, aes(x = Age)) + geom_histogram(bins = 30, fill = "blue", color = "black") + ggtitle("Distribution of Age") + custom_theme
p_spend <- ggplot(data, aes(x = Total.Spend)) + geom_histogram(bins = 30, fill = "green", color = "black") + ggtitle("Distribution of Total Spend") + custom_theme
p_items <- ggplot(data, aes(x = Items.Purchased)) + geom_histogram(bins = 30, fill = "red", color = "black") + ggtitle("Distribution of Items Purchased") + custom_theme
p_rating <- ggplot(data, aes(x = Average.Rating)) + geom_histogram(bins = 30, fill = "purple", color = "black") + ggtitle("Distribution of Average Rating") + custom_theme
p_days <- ggplot(data, aes(x = Days.Since.Last.Purchase)) + geom_histogram(bins = 30, fill = "orange", color = "black") + ggtitle("Days Since Last Purchase") + custom_theme

# Categorical variables
p_gender <- ggplot(data, aes(x = Gender)) + geom_bar(fill = "steelblue") + ggtitle("Gender Distribution") + custom_theme
p_membership <- ggplot(data, aes(x = Membership.Type)) + geom_bar(fill = "coral") + ggtitle("Membership Type Distribution") + custom_theme
p_satisfaction <- ggplot(data, aes(x = Satisfaction.Level)) + geom_bar(fill = "lightgreen") + ggtitle("Satisfaction Level Distribution") + custom_theme

# Bivariate Analysis
p_age_spend <- ggplot(data, aes(x = Age, y = Total.Spend)) + geom_point() + ggtitle("Age vs Total Spend") + custom_theme
p_items_spend <- ggplot(data, aes(x = Items.Purchased, y = Total.Spend)) + geom_point() + ggtitle("Items Purchased vs Total Spend") + custom_theme
p_membership_spend <- ggplot(data, aes(x = Membership.Type, y = Total.Spend, color = Membership.Type)) + geom_boxplot() + ggtitle("Membership Type vs Total Spend") + custom_theme
p_satisfaction_spend <- ggplot(data, aes(x = Satisfaction.Level, y = Total.Spend, color = Satisfaction.Level)) + geom_boxplot() + ggtitle("Satisfaction Level vs Total Spend") + custom_theme

# Bivariate Boxplots
p_gender_spend <- ggplot(data, aes(x = Gender, y = Total.Spend, color = Gender)) + geom_boxplot() + ggtitle("Gender vs Total Spend") + custom_theme
p_membership_items <- ggplot(data, aes(x = Membership.Type, y = Items.Purchased, color = Membership.Type)) + geom_boxplot() + ggtitle("Membership Type vs Items Purchased") + custom_theme
p_membership_rating <- ggplot(data, aes(x = Membership.Type, y = Average.Rating, color = Membership.Type)) + geom_boxplot() + ggtitle("Membership Type vs Average Rating") + custom_theme
p_satisfaction_rating <- ggplot(data, aes(x = Satisfaction.Level, y = Average.Rating, color = Satisfaction.Level)) + geom_boxplot() + ggtitle("Satisfaction Level vs Average Rating") + custom_theme

# Scatterplot Matrix (pairs)
numeric_data <- data %>% select_if(is.numeric)
pairs(numeric_data, col = "blue", main = "Scatterplot of Numeric Predictors")

# Display the plots
plot_list <- list(
  p_age, p_spend, p_items, p_rating, p_days,
  p_gender, p_membership, p_satisfaction,
  p_age_spend, p_items_spend, p_membership_spend, p_satisfaction_spend,
  p_gender_spend, p_membership_items, p_membership_rating, p_satisfaction_rating
)

for (plot in plot_list) {
  print(plot)
}


##################################
# Data Cleaning/Preparation
##################################
# Checking for missing values
missing_values <- colSums(is.na(data))
print(missing_values)

# Handling Missing Values
data$Satisfaction.Level[is.na(data$Satisfaction.Level)] <- names(sort(table(data$Satisfaction.Level), decreasing = TRUE))[1]

# Convert 'Gender' and 'Membership Type' to factors with ordinal encoding
data$Gender <- factor(data$Gender, levels = c("Male", "Female"), labels = c(1, 2))
data$Membership.Type <- factor(data$Membership.Type, levels = c("Bronze", "Silver", "Gold"), labels = c(1, 2, 3))

# Drop 'Customer ID' and 'City' columns
data <- data[, !names(data) %in% c("Customer.ID", "City")]

# Encode 'Discount Applied' as numeric (binary encoding)
data$Discount.Applied <- ifelse(data$Discount.Applied == "TRUE", 1, 0)

# Encode 'Satisfaction Level' as ordinal
data$Satisfaction.Level <- factor(data$Satisfaction.Level, levels = c("Unsatisfied", "Neutral", "Satisfied"), ordered = TRUE)

# Remove rows with any remaining NA values
data <- na.omit(data)

# View the transformed data
head(data)
summary(data)

# Check the distribution of 'Discount Applied'
table(data$Discount.Applied)


#####################
# Linear Regression
#####################
train <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.8, 0.2))

# Correlation matrix calculation on numeric data only
cor_matrix <- cor(data[train, sapply(data[train, ], is.numeric)], use = "pairwise.complete.obs")
print(cor_matrix)

# Fit linear regression model with interactions
model <- lm(Total.Spend ~ . + Age:Membership.Type  , data = data)
summary(model)


#########################
# REGRESSION
#########################

# To remove zero-variance columns for both numeric and factor variables
data <- data[, sapply(data, function(x) {
  if (is.numeric(x)) {
    return(var(x, na.rm = TRUE) != 0)
  } else if (is.factor(x)) {
    return(length(unique(x)) > 1)
  } else {
    return(TRUE)  
  }
})]      # Here "Gender" column is removed


########################
# Best Subset Selection
########################

#data <- na.omit(data)  

# Fit a full model using best subset selection
regfit.full <- regsubsets(Total.Spend ~ . -Items.Purchased - Average.Rating, data=data, nvmax=11)
summary(regfit.full)

# Model diagnostics and plotting
plot(regfit.full, scale="Cp")

# Cross-Validation for model evaluation
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(data), rep=TRUE)
test <- (!train)

# Initialize val.errors with the size 7
val.errors <- rep(NA, 7)  # Here the length matches the maximum model size we are testing

# Fit the best subset selection model on the training data
regfit.best <- regsubsets(Total.Spend ~ . - Items.Purchased - Average.Rating , data=data[train,], nvmax=7)

# Prepare the model matrix for the test set
test.mat <- model.matrix(Total.Spend ~ . - Items.Purchased - Average.Rating , data=data[test,])

# Calculate validation errors for each model size
for (i in 1:7) {
  # Get coefficients of the best model of size i
  coefi <- coef(regfit.best, id=i)
  
  pred_names <- names(coefi)
  pred <- test.mat[, pred_names, drop = FALSE] %*% coefi
  
  # Calculate mean squared error
  val.errors[i] <- mean((data$Total.Spend[test] - pred)^2, na.rm = TRUE)
}

print(val.errors)  

# Fit the model using all available training data
best_model <- glm(Total.Spend ~ . - Items.Purchased - Average.Rating, data = data[train,])
summary(best_model)
par(mfrow=c(2, 2))
plot(best_model)

# Fit the best subset selection model on the training data
best_model_index <- which.min(val.errors)

# Extracting the coefficients for the best model and prepare predictions
best_coef <- coef(regfit.best, id=best_model_index)
pred_names <- names(best_coef)
pred_names

# Make sure the test matrix 'test.mat' corresponds with the model variables
pred_full <- test.mat[, pred_names, drop = FALSE] %*% best_coef
residuals_best_full <- data$Total.Spend[test] - pred_full

par(mfrow=c(2,2))

# Residuals vs Fitted Values Plot
plot(residuals_best_full ~ pred_full, xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted")
abline(h = 0, col = "red")

# Normal Q-Q Plot
qqnorm(residuals_best_full, main = "Normal Q-Q Plot")
qqline(residuals_best_full)

# Scale-Location Plot
sqrt_abs_standard_residuals <- sqrt(abs(residuals_best_full))
plot(sqrt_abs_standard_residuals ~ pred_full, xlab = "Fitted Values", ylab = "Square Root of Standardized Residuals", main = "Scale-Location Plot")

# Residuals vs Leverage Plot
model_for_leverage <- lm(Total.Spend ~ . - Average.Rating - Items.Purchased, data = data[test,])

# Calculate hat values (leverage scores)
leverage_values <- hatvalues(model_for_leverage)

# Plot Residuals vs Leverage
plot(leverage_values, residuals_best_full, xlab = "Leverage", ylab = "Residuals", main = "Residuals vs Leverage")
abline(h = 2*sqrt(mean(sqrt_abs_standard_residuals)), col="red", lty=2)


#############################
# FWD SUBSET SELECTION
#############################

#data <- na.omit(data)  

# Reset the train-test split
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.7, 0.3))
test <- !train

# Fit the forward selection model on the training data
regfit.forward <- regsubsets(Total.Spend ~ . - Items.Purchased - Average.Rating, data=data[train,], nvmax=7, method="forward")

# Prepare the model matrix for the test set
test.mat <- model.matrix(Total.Spend ~ . - Items.Purchased - Average.Rating, data=data[test,])

# Calculate validation errors for each model size
val.errors <- rep(NA, 7)  # Adjust the length to match the number of models
for (i in 1:7) {
  coefi <- coef(regfit.forward, id=i)
  pred_names <- names(coefi)
  pred <- test.mat[, pred_names, drop = FALSE] %*% coefi
  val.errors[i] <- mean((data$Total.Spend[test] - pred)^2, na.rm = TRUE)
}

# Selecting the best model based on the lowest validation error
best_model_index <- which.min(val.errors)
best_coef <- coef(regfit.forward, id=best_model_index)

# Extract predictions for the best model
pred_full <- test.mat[, names(best_coef), drop = FALSE] %*% best_coef
residuals_best_full <- data$Total.Spend[test] - pred_full

# Plot diagnostics
par(mfrow=c(2,2))
plot(residuals_best_full ~ pred_full, xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted")
abline(h = 0, col = "red")
qqnorm(residuals_best_full)
qqline(residuals_best_full)
sqrt_abs_standard_residuals <- sqrt(abs(residuals_best_full))
plot(sqrt_abs_standard_residuals ~ pred_full, xlab = "Fitted Values", ylab = "Square Root of Standardized Residuals", main = "Scale-Location Plot")

# Leverage Model
model_for_leverage <- lm(Total.Spend ~ . - Average.Rating - Items.Purchased, data = data[test,])
leverage_values <- hatvalues(model_for_leverage)
plot(leverage_values, residuals_best_full, xlab = "Leverage", ylab = "Residuals", main = "Residuals vs Leverage")
abline(h = 2*sqrt(mean(sqrt_abs_standard_residuals)), col="red", lty=2)


#############################
# BACKWARD SELECTION
############################

# Splitting data into training and testing sets
set.seed(1)
train_indices <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.7, 0.3))
data_train <- data[train_indices, ]
data_test <- data[!train_indices, ]

# Initial full model including all predictors, excluding non-predictive or not needed columns explicitly
full_model <- lm(Total.Spend ~ .  - Average.Rating - Items.Purchased, data = data_train)

# Backward selection based on AIC
backward_model <- step(full_model, direction = "backward")

# Summarize the selected model
summary(backward_model)

# Validation
test_mat <- model.matrix(Total.Spend ~ . - Average.Rating - Items.Purchased, data = data_test)[, -1]

# Extract coefficients from the backward model
coefi <- coef(backward_model)

# Calculate predictions
pred_full <- test_mat %*% coefi
residuals_best_full <- data_test$Total.Spend - pred_full

# Diagnostic Plots
par(mfrow=c(1,2))
plot(residuals_best_full ~ pred_full, main="Residuals vs Fitted", xlab="Fitted Values", ylab="Residuals")
abline(h=0, col="red")
qqnorm(residuals_best_full, main="Normal Q-Q Plot")
qqline(residuals_best_full)
sqrt_abs_residuals <- sqrt(abs(residuals_best_full))
plot(sqrt_abs_residuals ~ pred_full, main="Scale-Location Plot", xlab="Fitted Values", ylab="Square Root of Standardized Residuals")

# Leverage Plot
model_for_leverage <- lm(Total.Spend ~ . - Average.Rating - Items.Purchased, data = data_test)
leverage_values <- hatvalues(model_for_leverage)
plot(leverage_values, residuals_best_full, main="Residuals vs Leverage", xlab="Leverage", ylab="Residuals")
abline(h=2*sqrt(mean(sqrt_abs_residuals)), col="red", lty=2)

# Prepare the model matrix
x <- model.matrix(Total.Spend ~ . - Average.Rating - Items.Purchased, data = data)  # -1 to exclude intercept in glmnet
y <- data$Total.Spend

# Create indices for cross-validation
set.seed(1)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
train <- x[train_index, , drop = FALSE]
test <- x[-train_index, , drop = FALSE]
y_train <- y[train_index]
y_test <- y[-train_index]

############################
# Ridge Shrinkage Method
############################
cv.ridge <- cv.glmnet(train, y_train, alpha = 0, family = "gaussian")
best.lambda.ridge <- cv.ridge$lambda.min
ridge.model <- glmnet(train, y_train, alpha = 0, lambda = best.lambda.ridge)

# Coefficients
ridge_coefs <- coef(ridge.model, s = best.lambda.ridge)
par(mfrow=c(1,1))
# Predictions and Residuals
pred_ridge <- predict(ridge.model, s = best.lambda.ridge, newx = test)
residuals_ridge <- y_test - pred_ridge
plot(cv.ridge)
print(cv.ridge$cvm[cv.ridge$lambda == cv.ridge$lambda.min])  # Minimum mean cross-validated error

############################
# Lasso Shrinkage Method
############################
cv.lasso <- cv.glmnet(train, y_train, alpha = 1, family = "gaussian")
best.lambda.lasso <- cv.lasso$lambda.min
lasso.model <- glmnet(train, y_train, alpha = 1, lambda = best.lambda.lasso)

# Coefficients
lasso_coefs <- coef(lasso.model, s = best.lambda.lasso)

# Predictions and Residuals
pred_lasso <- predict(lasso.model, s = best.lambda.lasso, newx = test)
residuals_lasso <- y_test - pred_lasso
plot(cv.lasso)

# Compute mean cross-validated test errors
cv.errors_ridge <- mean((y_test - pred_ridge)^2)
cv.errors_lasso <- mean((y_test - pred_lasso)^2)

# Print mean cross-validated test errors
print(paste("Ridge Regression:", cv.errors_ridge))
print(paste("Lasso Regression:", cv.errors_lasso))

# Determine the optimal model
optimal_errors <- c(Ridge = cv.errors_ridge, Lasso = cv.errors_lasso)
optimal_model <- names(optimal_errors)[which.min(optimal_errors)]
print(paste("Optimal Model:", optimal_model))


#######################################################
# CLASSIFICATION
######################################################
attach(data)
head(data)

# Encoding 'Satisfied' in one class(positive) and others in one class
data$Satisfaction.Level <- ifelse(data$Satisfaction.Level == "Satisfied", 1, 0)
# Convert to factor for logistic regression
data$Satisfaction.Level <- factor(data$Satisfaction.Level, levels = c(0, 1))

# Summary to check the conversion
summary(data$Satisfaction.Level)

# Visualizing Age vs Total Spend by Satisfaction Level
ggplot(data, aes(x = Age, y = Total.Spend, color = Satisfaction.Level)) + 
  geom_point() + 
  theme_minimal()

# Boxplot of Total Spend by Membership Type colored by Satisfaction Level
ggplot(data, aes(x = Membership.Type, y = Total.Spend, fill = Satisfaction.Level)) + 
  geom_boxplot() + 
  theme_minimal()

# Correlation plot
cor_matrix <- cor(data[,sapply(data, is.numeric)])
corrplot::corrplot(cor_matrix, method = "circle")

# Visualization of Age vs. Satisfaction Level
ggplot(data, aes(x = Age, fill = Satisfaction.Level)) + 
  geom_histogram(position = "dodge", bins = 30)


# Quick look at pairwise relationships for a subset of relevant predictors
pairs(~ Total.Spend + Age + Gender + Membership.Type + Items.Purchased + Average.Rating, data = data)

# Transformation of variables for better model fitting
par(mfrow=c(1,2))
plot(Age, Total.Spend)
abline(lm(Total.Spend ~ Age), col="red")  # Basic linear fit
cor(Age, Total.Spend)

# Applying a log transformation on Age to check for improvements
plot(log(Age), Total.Spend)
abline(lm(Total.Spend ~ log(Age)), col="red")
cor(log(Age), Total.Spend)

set.seed(123)  # For reproducibility

# Split data into training and testing sets
trainIndex <- createDataPartition(data$Satisfaction.Level, p = 0.8, list = FALSE)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]


###########################
# LOGISTIC REGRESSION
###########################

# Fit logistic regression model using training data
glm.fit <- glm(Satisfaction.Level ~ Gender + Age + Membership.Type + 
                 Total.Spend + Days.Since.Last.Purchase, 
               family = binomial, 
               data = trainData)

# More robust logistic regression with tighter convergence criteria
glm.fit1 <- glm(Satisfaction.Level ~ Gender + Age + Membership.Type + 
                  Total.Spend + Days.Since.Last.Purchase, 
                family = binomial, 
                data = trainData,
                control = glm.control(epsilon = 1e-8, maxit = 50))

# Using brglm for potentially better handling of separation issues
glm.fit.brglm <- brglm(Satisfaction.Level ~ Gender + Age + Membership.Type + 
                         Total.Spend + Days.Since.Last.Purchase, 
                       family = binomial, data = trainData)

# Summary of the logistic regression models
summary(glm.fit)
summary(glm.fit1)
summary(glm.fit.brglm)

# Display Cook's Distance to identify influential cases
par(mfrow=c(1, 3))
plot(cooks.distance(glm.fit), type="h", main="Cook's Distance")
plot(cooks.distance(glm.fit1), type="h", main="Cook's Distance")
plot(cooks.distance(glm.fit.brglm), type="h", main="Cook's Distance")

# Calculate coefficients, standard errors, and confidence intervals
coefficients <- coef(glm.fit.brglm)
std.errors <- summary(glm.fit.brglm)$coefficients[, "Std. Error"]
lower_bounds <- coefficients - 1.96 * std.errors
upper_bounds <- coefficients + 1.96 * std.errors
wald.conf.int <- data.frame(
  Lower = exp(lower_bounds),
  Estimate = exp(coefficients),
  Upper = exp(upper_bounds)
)

# Print the confidence intervals for the odds ratios
print(wald.conf.int)

# Setting up bootstrap with R's 'boot' package
boot_model <- function(data, indices) {
  d <- data[indices,]  # resample with replacement
  fit <- brglm(Satisfaction.Level ~ Gender + Age + Membership.Type + 
                 Total.Spend + Days.Since.Last.Purchase, 
               data = d, family = binomial)
  return(coef(fit))
}
set.seed(123)  # for reproducibility
bootstrap_results <- boot(trainData, boot_model, R = 1000)
boot.ci(bootstrap_results, type = "bca")

# Plot the bootstrap distributions
par(mfrow=c(2, 4)) # Adjust this based on the number of predictors
for(i in 1:ncol(bootstrap_results$t)){
  hist(bootstrap_results$t[, i], main=names(bootstrap_results$t[, i]), xlab="Coefficient Estimate")
}

# Predict probabilities and labels for test data
glm.probs <- predict(glm.fit.brglm, testData, type = "response")
glm.pred <- factor(ifelse(glm.probs > 0.5, "Satisfied", "Unsatisfied"), levels = c("Satisfied", "Unsatisfied"))

# Define cross-validation control
control <- trainControl(method="cv", number=10)

# Compute cross-validated test error
cv_model <- train(Satisfaction.Level ~ Gender + Age + Membership.Type + 
                    Total.Spend + Days.Since.Last.Purchase, 
                  data=trainData, method="glm", family="binomial", trControl=control)
cv_error <- 1 - max(cv_model$results$Accuracy)
print(paste("Cross-validated test error:", cv_error))

# Predict probabilities and labels for test data
glm.probs <- predict(glm.fit, newdata = testData, type = "response")
glm.pred <- factor(ifelse(glm.probs > 0.5, "1", "0"), levels = c("1", "0"))
testData$Satisfaction.Level <- factor(testData$Satisfaction.Level, levels = c("1", "0"))

# Create confusion matrix
confusionMatrix(table(glm.pred, testData$Satisfaction.Level))

# Calculate specificity and sensitivity
conf_matrix <- table(glm.pred, testData$Satisfaction.Level)


#########
# LDA
#########

# Build LDA model
lda_model <- lda(Satisfaction.Level ~ Gender + Age + Membership.Type + 
                   Total.Spend + Days.Since.Last.Purchase, data = trainData)

# Summary of the model
print(lda_model)

# Cross-validation to estimate model performance
set.seed(123)  # for reproducibility
control <- trainControl(method = "cv", number = 10, savePredictions = "final")
cv_model <- train(Satisfaction.Level ~ Gender + Age + Membership.Type + 
                    Total.Spend + Days.Since.Last.Purchase, 
                  data = trainData, 
                  method = "lda", 
                  trControl = control)

# Extract cross-validated accuracy and compute test error
cv_accuracy <- max(cv_model$results$Accuracy)
cv_error <- 1 - cv_accuracy

# Predict on test data using the lda model
predictions <- predict(lda_model, testData)

# Create confusion matrix
conf_matrix <- table(Predicted = predictions$class, Actual = testData$Satisfaction.Level)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate specificity and sensitivity
specificity <- conf_matrix[2,2] / sum(conf_matrix[2, ])
sensitivity <- conf_matrix[1,1] / sum(conf_matrix[1, ])

# Print best values
cat("Cross-validated test error:", cv_error, "\t Sensitivity:", sensitivity, "\t Specificity:", specificity, "\n")


#########
# KNN
#########

# Set seed for reproducibility
set.seed(1)

# Prepare the data
predictors <- c("Age", "Total.Spend", "Days.Since.Last.Purchase")
train.X <- trainData[, predictors]
test.X <- testData[, predictors]
train.Direction <- trainData$Satisfaction.Level

# Initialize variables to track the best metrics
best_k <- 0
best_accuracy <- 0
best_sensitivity <- 0
best_specificity <- 0
cv_error_best <- Inf

# Loop through different values of K
for(k in 1:10) {
  # Apply kNN
  knn.pred <- knn(train.X, test.X, train.Direction, k = k)
  
  # Create confusion matrix
  conf_matrix <- table(Predicted = knn.pred, Actual = testData$Satisfaction.Level)
  
  # Calculate accuracy
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  
  # Calculate sensitivity (True Positive Rate)
  sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  
  # Calculate specificity (True Negative Rate)
  specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
  
  # Cross-validated test error
  cv_error <- 1 - accuracy
  
  # Update best values if current model is better
  if(accuracy > best_accuracy) {
    best_accuracy <- accuracy
    best_k <- k
    best_sensitivity <- sensitivity
    best_specificity <- specificity
    cv_error_best <- cv_error
  }
  # Print results
  cat("K =", k, "\tAccuracy:", accuracy, "\tSensitivity:", sensitivity, "\tSpecificity:", specificity, "\tCV Error:", cv_error, "\n")
}

# Print best values
cat("Best K:", best_k, "\tBest Accuracy:", best_accuracy, "\tBest Sensitivity:", best_sensitivity, "\tBest Specificity:", best_specificity, "\tBest CV Error:", cv_error_best, "\n")

