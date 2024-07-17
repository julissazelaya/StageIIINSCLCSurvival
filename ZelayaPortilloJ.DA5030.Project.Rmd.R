#' ---
#' title: "Stage III NSCLC Survival Prediction"
#' author: "Julissa Zelaya Portillo"
#' date: "Fall 2023"
#' subtitle: "DA5030"
#' output:
#'   pdf_document: default
#'   html_document:
#'     df_print: paged
#'     toc: true
#'     toc_float: true
#' ---
#' 
#' ## Business Understanding
#' A physician's value is in accurately identifying and diagnosing patients
#' based on reported metrics. In the case of 548 patients, data is obtained
#' from the following clinical investigation to build a predictive model: "A Validated
#' Prediction Model for Overall Survival From Stage III Non-Small Cell Lung Cancer:
#' Toward Survival Prediction for Individual Patients" (Oberije, et al., 2015). The
#' original study provides a cox regression model based on 22 clinical variables. 
#' 
#' It is the aim of this markdown to implement various classification machine
#' learning models. Unlike the original model, this does not include a time
#' component, but may provide useful insight on model comparisons. More importantly,
#' this may also provide an additional perspective on whether a patient is marked
#' for near-death or survival.
#' 
## ----library, echo=FALSE, message=FALSE, warning=FALSE, include=FALSE------------------------------------------------------------------------------------------------------------------
# List of packages to check and install
packages_to_install <- c("dplyr", "ggplot2", "fastDummies", "mice","stats",
                         "caret", "kernlab", "class", "pROC", "ipred", 
                         "randomForest")

# Function to check and install packages
install_packages <- function(packages) {
  for (package in packages) {
    if (!requireNamespace(package, quietly = TRUE)) {
      # Unload the package if it is loaded
      if (package %in% rownames(installed.packages())) detach(paste0("package:"
                                                                     , package),
                                                              unload = TRUE)
      # Install the package
      install.packages(package, dependencies = TRUE)
    }
    # Load the package
    library(package, character.only = TRUE)
  }
}

# Call the function to install and load packages
install_packages(packages_to_install)

#' 
#' 
#' ## Data Understanding
#' ### Data Exploration
#' The original NSCLC data is formatted as a `.csv` file and is acquired through
#' its open-source URL address. 
#' 
## ----data_acquisition------------------------------------------------------------------------------------------------------------------------------------------------------------------
url <- ("https://www.cancerdata.org/system/files/publications/Stage3_anonymized.csv")
cancer <- read.csv(url, sep = ";") # source CSV is not comma-separated
# Check the structure of the data frame
str(cancer)

#' 
#' The structure of the original data shows a data frame with the expected 548
#' patients. The additional feature for `study_id` will need to be removed to focus
#' on the 22 clinical variables. It also becomes known that there are missing values
#' within the data that will need to be addressed later on. The immediate concern
#' is that there are features that use commas instead of decimals. These features
#' incorrectly have a character data type instead of numeric. This formatting issue
#' will need to be addressed before exploratory visualizations can be performed.
#' 
#' Commas are replaced with periods in the eight listed character features. This is
#' to accurately represent decimals (floats). All integers are then converted to
#' factors to distinguish the presence of categorical features. 
#' 
## ----exploration_data_types------------------------------------------------------------------------------------------------------------------------------------------------------------
# Remove first column as ID is not necessary
cancer <- cancer[, -1]
# Remove duplicate feature as survival month equals the survival year
cancer$survmonth <- NULL

# List character features data types that should be numeric
chr_to_numeric <- c("age", "bmi", "fev1pc_t0", "eqd2", "ott", "gtv1",
                    "tumorload_total", "survyear")

# Replace commas with periods
cancer <- cancer %>%
  mutate(across(all_of(chr_to_numeric), ~gsub(",", ".", .)))

# Convert the specified features to numeric
cancer <- cancer %>%
  mutate(across(all_of(chr_to_numeric), as.numeric))

# Identify integer columns
integer_cols <- sapply(cancer, is.integer)

# Factorize integer variables
cancer[, integer_cols] <- lapply(cancer[, integer_cols], as.factor)

#' 
#' All incorrect data types are now numeric. It is noted that categorical features
#' have label encoding applied. These features are labeled and factored to make
#' interpretation of these variables easier later on. 
#' 
#' ### Detection of Outliers for Continuous Features
#' Based on the output below, is evident that there are significant
#' outliers--especially from the two tumor related variables. The question is
#' whether this is due to the exponential growth that occurs in tumor cells.
#' 
## ----exploration_boxplot---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a data frame with continuous features from `cancer` data frame
cancer_continuous <- data.frame(cancer$age, cancer$bmi, cancer$fev1pc_t0,
                                cancer$eqd2,cancer$ott, cancer$gtv1, 
                                cancer$tumorload_total, cancer$survyear)

# Specify window for eight variables
par(mfrow = c(2, 4)) 

# Create box plots for each continuous variable
for (variable in colnames(cancer_continuous)) {
  boxplot(cancer_continuous[[variable]], main = variable, col = "red")
}

#' 
#' A z-score method is implemented to better identify the extent of how many
#' outliers there are. A non-conservative approach is taken considering that the
#' nature of cancer cells may cause unexpected extremes.
#' 
## ----exploration_zscore----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to detect outliers using Z-scores
detect_outliers <- function(x, threshold = 3) {
  x_no_na <- na.omit(x)  # Remove missing values
  z_scores <- scale(x_no_na)
  outliers <- abs(z_scores) > threshold
  return(sum(outliers))
}

# Apply the function to each variable
outlier_counts <- sapply(cancer_continuous, detect_outliers)

# Display the number of outliers for each variable
print(outlier_counts)

#' In addition to the previously noted outliers, there are outliers including
#' survival years and BMI. It is hard to determine the value of these
#' outlier data point so this information is noted and will be considered during
#' data cleaning and shaping. 
#' 
#' ### Evaluation of Distribution
#' Further insight of the data distribution can be gained from density plots.
#' 
## ----exploration_plot------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Show all plots together
par(mfrow = c(2, 4))

# Create density plots for subset of continuous features
for (i in 1:8) {
  # Calculate density
  dens <- density(cancer_continuous[, i], na.rm = TRUE)
  
  # Plot density
  plot(dens, main = names(cancer_continuous)[i],
       xlab = names(cancer_continuous)[i], col = "red", lwd = 2)
}

#' 
#' The density plots show that most continuous features are positively skewed. These
#' right-skewed features may require transformations to redistribute data. This
#' can be confirmed through the use of a Shapiro-Wilk Test. 
#' 
## ----exploration_shapiro---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to perform Shapiro-Wilk test
shapiro_function <- function(x) {
  result <- shapiro.test(x)
  return(c(result$statistic, "P Value" = result$p.value)) # return p-value
}

# Apply the function to each continuous variable
shapiro_results <- sapply(cancer_continuous, shapiro_function)

# Create a data frame for display
shapiro_df <- data.frame(t(shapiro_results))
shapiro_df

#' 
#' A p-value less than the chosen significance level (0.05), indicates strong
#' evidence to reject the null hypothesis. Each p-value shown above has a low
#' p-value meaning all these features do not follow a normal distribution. This
#' supports the visually skewed data plot from the output above. The only exception
#' is the `fev1pc_t0` (Forced Expiratory Volume). Regardless, it is understood
#' that feature transformations will need to be made. 
#' 
#' All continuous features have now been evaluated for outliers and overall data
#' distribution. It is now important to have an understanding of the distribution of
#' the categorical features. 
#' 
#' The frequency of each level within each categorical feature is shown in
#' the frequency table below.
#' 
## ----exploration_categorical_frequency-------------------------------------------------------------------------------------------------------------------------------------------------
# Specify the categorical variables
cancer_categorical <- c("gender", "who3g", "dumsmok2", "t_ct_loc", "hist4g",
                        "countpet_all6g", "countpet_mediast6g",
                        "tstage", "nstage", "stage", "timing",
                        "group", "yearrt")

# Create and print percentage tables for each categorical variable
for (variable in cancer_categorical) {
  cat("Percentage Distribution of", variable, ":\n")
  print(prop.table(table(cancer[[variable]])) * 100)
  cat("\n")
}

#' 
#' Two features that initially seem to have skewed data are based on gender and
#' smoker status (`dumsmok2`). This clinical data is acquired from 70% of males and
#' 60% of non-smokers.
#' 
#' A frequency table is now obtained for the target feature.
#' 
## ----exploration_target_frequency, warning=FALSE, message=FALSE------------------------------------------------------------------------------------------------------------------------
# Rename 0 for "Alive" and 1 for "Dead"
cancer$deadstat <- factor(cancer$deadstat, levels = c(0, 1), 
                        labels = c("Alive", "Dead"))

# Display percentage of each patient outcome
round(prop.table(table(cancer$deadstat)) * 100, digits = 0)

#' 
#' 83% of outcomes classify death. This is very important to note as this majority
#' may introduce a bias when training a classification model. In terms of evaluation
#' metrics, accuracy alone will not be sufficient. Precision, recall, F1 score,
#' and/or area under the ROC curve will need to be used later on.
#' 
#' ### Correlation, Collinearity, and Chi-Squared Analysis
#' It is now important to determine the effect of collinearity. This is the
#' correlation between predictor variables that would create what is considered 
#' "noise" in a data model. The scatterplot matrix created below presents a visual
#' interpretation of this. 
#' 
## ----exploration_correlation_scatterplot-----------------------------------------------------------------------------------------------------------------------------------------------
# Create scatterplot matrix of continuous features
pairs(cancer_continuous)

#' 
#' A linear relationship is noted between gross tumor volume and the total tumor
#' load. This is expected because of the overlapping nature of these features.
#' 
#' These relationships can be confirmed through a matrix of Pearson's correlation
#' values. This would highlight individual relationships or suggest collinearity
#' between any continuous variables. 
#' 
## ----exploration_correlation-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the correlation matrix, skipping missing values
cor_matrix <- cor(cancer_continuous, use = "complete.obs", method = "pearson")
cor_matrix

#' 
#' The correlation matrix confirms a strong relationship between gross tumor
#' volume and tumor load total with a value of 0.85. This will need to be addressed
#' during data preparation as this could introduce noise into our model.
#' 
#' As the correlation matrix accounts for continuous variables, a chi-squared
#' analysis will be performed for the remaining categorical variables. 
#' 
## ----exploration_chi_squared-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Perform the chi-squared test on the categorical variables
chi_results <- lapply(cancer[, cancer_categorical], function(x) {
  chisq.test(table(x))
})
chi_results

#' 
#' If the p-value is less than the chosen significance level (0.05), the null
#' hypothesis is rejected. A null hypothesis provides evidence of an association
#' between variables. Based on the output above, these features are not independent
#' and there is an association between them.
#' 
#' 
#' ## Data Preparation
#' With an understanding of the NSCLC cancer data set, data cleaning and shaping
#' will be performed for better model constructions.This encompasses the imputation
#' of missing data, normalization/standardization, dummy codes, feature
#' transformation, PCA, and feature engineering. 
#' 
#' ### Identification of Missing Values
#' It is necessary to first determine the extent of missing values in the working
#' data set. The count of missing values are presented by feature to compare to
#' the overall 548 observations. For instance, the BMI feature has 179 missing
#' values that corresponds to 33% of the feature data being missing.
#' 
## ----cleaning_identify_missing---------------------------------------------------------------------------------------------------------------------------------------------------------
# Identify missing values in the entire data set per feature
missing_values <- sapply(cancer, function(x) sum(is.na(x)))
missing_values

#' 
#' ### Data Imputation of Missing Data
#' It is a common method to input missing data with the mean, median, or mode of
#' that value's feature, but there are a considerate amount of missing values here.
#' A more sophisticated method of imputing randomly missing data is through the use
#' of the `mice` package. The `mice` package will use Predictive Mean Matching to
#' fill in missing values by drawing observations from other values close-by and
#' making its own prediction. This will retain the original distribution of the
#' data set.
#' 
## ----cleaning_missing_imputation, include = FALSE, warning=FALSE-----------------------------------------------------------------------------------------------------------------------
# Impute missing values using mice
imputed_data <- mice(cancer, m = 5, method = 'pmm', seed = 123)

# Create new working data frame with imputed data
lung <- complete(imputed_data)

#' 
#' Missing data has been imputed into the new `lung` data frame with the Predictive
#' Mean Matching (pmm) method, as previously mentioned. The `lung` data frame is
#' checked again for missing values and to ensure that there are no drastic changes
#' within each feature's data distribution. 
#' 
#' ### Feature Engineering: New Derived Features
#' With the confirmation that there are no missing values, we return to the issue
#' of collinearity between the gross tumor volume and total tumor load. The new
#' derived feature will represent the proportion of total tumor load that is
#' contributed by the gross tumor volume. 
#' 
## ----cleaning_collinearity_interaction-------------------------------------------------------------------------------------------------------------------------------------------------
# Rewrite is done to preserve order of columns
lung$gtv1 <- lung$gtv1 / lung$tumorload_total

# Rename the `gtv1` column to `gtv1_total`
lung <- rename(lung, gtv1_total = gtv1)

# Remove original `tumorload_total` to avoid duplicates
lung <- subset(lung, select = -c(tumorload_total))

#' 
#' ### Normalization/Standardization of Data
#' When comparing normalization versus standardization of data, it is important to 
#' consider the distribution of the working data set and the desired machine
#' learning models. While normalization is highly affected by outliers, the use of
#' standardization would create many negative values. This is because the original
#' data has a large range in values. This is noted by the presence of many outliers
#' and a non-normal distribution.
#' 
#' It is best to use Min-Max scaling (normalization) on continuous features as this
#' keeps the data to a specific range between 0 and 1. 
#' 
## ----cleaning_normalization------------------------------------------------------------------------------------------------------------------------------------------------------------
# Identify numeric columns
lung_numeric <- sapply(lung, is.numeric)

# Min-Max scaling
lung[, lung_numeric] <- scale(lung[, lung_numeric], center = FALSE)

#' 
#' Standardization is confirmed with all numerical values being between 0 and 1.
#' 
#' ### Dummy Codes
#' Nominal features are identified to appropriately apply dummy encoding. The
#' original data already established label encoding for ordinal features so they
#' will be excluded to preserve their ranking feature. This will be beneficial when
#' working with various models that may need easier distinction between rankings or
#' categorical features.
#' 
## ----cleaning_dummy--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Establish categorical, nominal variables that need dummy encoding
lung_categorical_nominal <- c("gender", "dumsmok2", "t_ct_loc", "hist4g")

# Dummy encode nominal variables
lung <- dummy_cols(lung, select_columns = lung_categorical_nominal)

# Remove the original columns after dummy encoding
lung <- lung[, !names(lung) %in% lung_categorical_nominal]

#' 
#' The original columns are then removed to avoid duplicates that would cause
#' collinearity.
#' 
#' ### Transformation of Features to Adjust Distribution
#' The working data set has numerical features with high skewness and non-normal
#' distribution (505-508 Distribution). This may be caused by outliers, highly 
#' exponential distributions, etc. Therefore, the logarithmic transformation is best.
#' 
## ----cleaning_transformation-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Identify numeric columns programmatically
lung_numeric <- names(Filter(is.numeric, lung))

# Apply log transformation to numeric variables
lung[, lung_numeric] <- log(lung[, lung_numeric] + 1)  # Adding 1 to avoid log(0)

#' 
#' If we were to now look at the distribution of one of the numerical features, we
#' should now expect to see a more normal distribution.
#' 
## ----cleaning_transformation_hist------------------------------------------------------------------------------------------------------------------------------------------------------
# Show plots side by side
par(mfrow = c(1, 2))

# Compare histogram plots before and after data processing
hist(cancer$survyear)
hist(lung$survyear)

#' 
#' As seen above, the survival year feature is compared from its the original raw
#' data and the current working data set. It is no longer positively skewed and has
#' a normal distribution.
#' 
#' ### Creation of Training and Validation Subsets
#' It is important to note that Principal Component Analysis and Feature Engineering
#' will be performed on the working data. Before beginning those processes, the
#' data needs to be split into training and testing data sets. If we were to obtain
#' PCA components of the data at once, the test data would get "leaked" into the
#' training data. This defeats the purpose of building a model that could be
#' generalized to other data sets.
#' 
#' The data set, `lung` will now be split into training and testing data sets.
#' 
## ----model_split data------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set seed for reproducibility
set.seed(123)  

# Perform a random split
splitIndex <- createDataPartition(lung$deadstat, p = 0.8, list = FALSE)
train_data <- lung[splitIndex, ]
test_data <- lung[-splitIndex, ]

#' 
#' ### Identification of Principal Components (PCA)
#' Principal Component Analysis reduces the dimension of a data set. Considering
#' the large numbers of features in the `lung` data, this would simplify analysis
#' while preserving variance.
#' 
#' The PCA model is created and the correlation between the features and principal
#' components are shown.
#' 
## ----cleaning_pca_model----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create PCA model
pca_model <- prcomp(train_data[, lung_numeric], center = TRUE, scale = TRUE)

# Apply PCA to the training set
train_data_pca <- predict(pca_model, train_data[, lung_numeric])
# Use the same PCA parameters from the training set on the testing set
test_data_pca <- predict(pca_model, test_data[, lung_numeric])

# Identify the principal components
principal_components <- pca_model$rotation

# Display the first few principal components
head(principal_components)

#' 
#' In the interpretation of the Principal Components, we narrow in on the `age`
#' variable under PC1. The value of 0.305 shows a weak, inverse relationship between
#' `age` and PC1.
#' 
#' The principal components are visually identified below. The arrows represent the
#' principal components (vectors) and their relationship with the original features.
#' Longer arrows would indicate features with higher contributions to the principal
#' components and the angles between arrows provide insights into feature
#' correlations.
#' 
## ----cleaning_pca_plot-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot loadings
biplot(pca_model)

#' 
#' Given that all vectors vary in directions suggest a lack of correlation between
#' the other features. This is no surprise as duplicate and overlapping features
#' have previously been removed in this markdown.
#' 
#' To make a final determination on PCA results, the proportion of variance
#' explained by the principal component is calculated.
#' 
## ----cleaning_pca_variance-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the proportion of variance explained by each principal component
proportion_of_variance <- pca_model$sdev^2 / sum(pca_model$sdev^2)
head(proportion_of_variance)

#' 
#' The first principal components captures roughly 7.8% of the total variance in
#' the data. Based on this low value, it is determined not to use the PCA.
#' 
#' 
#' ## Modeling
#' Various predictive models are built to classify survival of patients with 
#' Stage III Non-Small Cell Lung Cancer. In theory, this could alert physicians to
#' more urgent cases of NSCLC. 
#' 
#' ### Creation of Logistic Regression Model with Proper Data Encoding
#' First, a Logistic Regression model is made with the `glm()` function.
#' 
## ----model_A, warning=FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create logistic regression model
logistic_model <- glm(deadstat ~., data = train_data, 
                      family = "binomial", maxit = 1000)
logistic_predictions <- predict(logistic_model,
                                newdata = test_data, type = "response")

#' 
#' ### Creation of kNN Analysis with Proper Data Encoding
#' This is followed by the creation of a k-NN analysis with the `knn()` function.
#' It should be noted that there is a question of the appropriateness of the k-NN
#' model considering the data is high dimensional. Throughout the use
#' of the k-NN model, parameter tuning will need to be done to mitigate performance, 
#' data structure, and other logistical issues of using k-NN. 
#' 
## ----model_B, warning=FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------
# Assign "deadstat" is the target variable
labels_train <- train_data$deadstat
labels_test <- test_data$deadstat

# Exclude labels from the training and testing data
train_data_excluded_labels <- subset(train_data, select = -deadstat)
test_data_excluded_labels <- subset(test_data, select = -deadstat)

# Set seed for reproducibility
set.seed(123)

# Create the k-NN model using knn function
knn_model <- knn(train = train_data_excluded_labels, 
                 test = test_data_excluded_labels, cl = labels_train, k = 5)

#' 
#' ### Creation of Support Vector Machine (SVM) with Proper Data Encoding
#' And the last model being a Support Vector Machine with the `ksvm()` function.
#' 
## ----model_C, include=FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create an SVM model
svm_model <- ksvm(deadstat ~ ., data = train_data, 
                  type = "C-svc",  # C-classification
                  kernel = "vanilladot")  # Linear kernel
svm_predictions <- predict(svm_model, test_data)

#' 
#' ## Evaluation
#' Now that we have our three prediction models, the fit of each model is evaluated
#' with proper holdout methods. This compares the predicted outcomes with the actual
#' outcomes.
#' 
#' In working with a complex data set, it is best to not only rely on a reported
#' accuracy value. Precision (positive predictive value) is calculated as the
#' number of true positives divided by the sum of true positives and false positives.
#' Recall (sensitivity) is calculated as the number of true positives divided by the
#' sum of true positives and false negatives. F1 Score is then the mean of precision
#' and recall. An advantage of calculating the F1 score is that small precision or
#' recall will result in lower overall score. The F1 score helps to balance the two.
#' This is then followed with the reported area under the ROC curve (AUC) that adds
#' another metric to suggest whether the classifier has predictive value. 
#' 
#' ### Evaluation of Fit of Models with Holdout Method
#' There are various methods outlined below to evaluate the fit of the three models.
#' A confusion matrix compares the accurate predictions versus false predictions.
#' This is then followed by the accuracy, precision, recall, F1 score and AUC.
#' 
## ----evaulation_holdout_log, warning=FALSE, message=FALSE------------------------------------------------------------------------------------------------------------------------------
# Reproducibility
set.seed(123)

# Evaluate the logistic regression model
predicted_log <- ifelse(logistic_predictions > 0.5, "Dead", "Alive")
confusion_matrix_log <- confusionMatrix(test_data$deadstat, 
                                        as.factor(predicted_log))
confusion_matrix_log

# Extract F1 
f1_log <- confusion_matrix_log$byClass["F1"]
f1_log

# Create a ROC curve
roc_curve_log <- roc(as.numeric(test_data$deadstat == "Dead"), 
                     as.numeric(predicted_log == "Dead"))
# Calculate ROC-AUC
roc_auc_log <- auc(roc_curve_log)
roc_auc_log

#' 
#' The logistic regression model has a good performance/predictive value. Accuracy
#' and AUC is reported at a high 91.7% and 86% with lower values in precision
#' and recall (74% and 78%). In a practical evaluation, the confusion matrix shows
#' four false negatives that would have real-life implications of suggesting that
#' the patients have more years of survival when in reality, they're marked for death.
#' 
#' We move on to evaluate the k-NN classifier.
#' 
## ----evaluation_holdout_knn, message=FALSE---------------------------------------------------------------------------------------------------------------------------------------------
# Create a confusion matrix
confusion_matrix_knn <- confusionMatrix(knn_model, test_data$deadstat)
confusion_matrix_knn

# Extract F1 
f1_knn <- confusion_matrix_knn$byClass["F1"]
f1_knn

# Create a ROC curve for k-NN
roc_curve_knn <- roc(as.numeric(test_data$deadstat == "Dead"), 
                     as.numeric(knn_model == "Dead"))

# Calculate ROC-AUC for k-NN
roc_auc_knn <- auc(roc_curve_knn)
roc_auc_knn

#' 
#' This evaluation highlights the importance of not only using the reported accuracy,
#' but to take other performance measures into account. While the k-NN classification
#' has a high accuracy of 86%, it performs terribly when other metrics are
#' considered. For instance, an F1 value of 34.8% indicates that this is a low
#' performing model. The AUC of 0.606 then indicates that this model has no ability
#' to discriminate between outcomes. To take this into a real-world consideration, 
#' only four patients were accurately reported as being alive. This shows that the
#' model is more likely to classify each patient as being near-death--regardless
#' of how wrong it is. Note that we previously mentioned the possibility of this
#' when seeing the large bias of "Death" outcomes in the original data. 
#' 
#' We continue to move on to the Support Vector Machine model evaluation. 
#' 
## ----evaluation_holdout_svm, message=FALSE---------------------------------------------------------------------------------------------------------------------------------------------
# Create a confusion matrix using caret
confusion_matrix_svm <- confusionMatrix(as.factor(svm_predictions), 
                                        as.factor(test_data$deadstat))
confusion_matrix_svm

# Extract F1 
f1_svm <- confusion_matrix_svm$byClass["F1"]
f1_svm

# Calculate ROC-AUC
roc_curve_svm <- roc(as.numeric(test_data$deadstat == "Dead"), 
                     as.numeric(svm_predictions == "Dead"))
roc_auc_svm <- auc(roc_curve_svm)
roc_auc_svm

#' 
#' All metrics above show an incredibly high performing model. The confusion matrix
#' shows six total instances (0.06%) of incorrect predictions. Accuracy has a value
#' of 94.5%, AUC is 0.90 and F1 score and related values are 83%. It is still
#' important to note that three false negatives occurred. Therefore, three patients
#' would have been told that they still had more time until their death when they
#' would have actually been on the verge of death. Both sides of these performance
#' metrics are important to consider if using this model.
#' 
#' ### Evaluation with K-Fold Cross-Validation
#' Instead of evaluating each model based on a split with the previous holdout
#' method, a k-fold Cross Validation is now used. With the k-Fold Cross Validation
#' method,a single parameter called k refers to the number of groups that a given
#' data sample is to be split into. Note the tuning of hyperparameters for each
#' three models. 
#' 
## ----evaluation_kfold_log, warning=FALSE-----------------------------------------------------------------------------------------------------------------------------------------------
# Reproducibility
set.seed(123)

# Define cross-validation settings
ctrl <- trainControl(method = "cv", number = 10, # 10-fold cross-validation
                     summaryFunction = twoClassSummary, classProbs = TRUE)  

# Train the logistic regression model using k-fold cross-validation
logistic_cv_results <- train(
  deadstat ~ .,
  data = lung,
  method = "glm",
  trControl = ctrl
)
logistic_cv_results

# Make predictions
predictions_cv_logistic <- predict(logistic_cv_results, 
                                   newdata = lung, type = "raw")

# Class labels for the entire data set
actual_labels_logistic <- lung$deadstat

# Make sure both predicted and actual labels are factors with the same levels
predictions_cv_logistic <- factor(predictions_cv_logistic, 
                                  levels = levels(actual_labels_logistic))

# Create a confusion matrix for logistic regression
conf_matrix_cv_logistic <- confusionMatrix(predictions_cv_logistic, 
                                           actual_labels_logistic)
conf_matrix_cv_logistic

#' 
#' The logistic regression model via the k-Fold Cross Validation shows a high
#' performance with an ROC of 0.918 and recall value of 0.807. What is very
#' surprising is the confusion matrix showing a perfect model with 100% accuracy, 
#' recall and precision. All scores overall point to this being a great predictive
#' model.
#' 
#' Below, the k-NN classifier is similarly split.
#' 
## ----evaluation_kfold_knn--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reproducibility
set.seed(123)

# Set control parameter
knn_ctrl <- trainControl(method = "cv", number = 10)

# Define the hyperparameter grid for k-NN
knn_grid <- expand.grid(k = seq(1, 20, by = 1))

# Train the k-NN model using k-fold cross-validation
knn_cv_results <- train(
  deadstat ~.,
  data = lung,
  method = "knn",
  trControl = knn_ctrl,
  tuneLength = 10,
  tuneGrid = knn_grid
)
knn_cv_results

# Make predictions on the entire dataset
predictions_cv_knn <- predict(knn_cv_results, newdata = lung)

# Class labels for the entire dataset
actual_labels_knn <- lung$deadstat  # Adjust this based on your actual data

# Create a confusion matrix for k-NN
conf_matrix_cv_knn <- confusionMatrix(predictions_cv_knn, actual_labels_knn)
conf_matrix_cv_knn

#' 
#' The optimal model shows an accuracy of 85.4% with a high precision of 94.7%, but
#' an alarmingly low recall of 20%. While the confusion matrix has only false
#' negative, 80% of patients that are told they'd die soon actually live longer. 
#' This is not a great statistic but to some, it may be better to report this 
#' false positive (false death) instead of false negative (false survival).
#' 
#' 
#' We continue on to the SVM model.
#' 
## ----evaluation_kfold_svm, warning=FALSE-----------------------------------------------------------------------------------------------------------------------------------------------
# Train the SVM model using k-fold cross-validation
svm_cv_results <- train(
  deadstat ~.,
  data = lung,
  method = "svmLinear",
  trControl = ctrl,
  metric = "ROC"
)
svm_cv_results

# Make predictions
predictions_cv_svm <- predict(svm_cv_results, newdata = lung, type = "raw")

# Use class labels for the entire dataset
actual_labels_cv_svm <- lung$deadstat

# Create a confusion matrix
conf_matrix_cv_svm <- confusionMatrix(predictions_cv_svm, actual_labels_cv_svm)
conf_matrix_cv_svm


#' 
#' The SVM model shows a relatively good performance with an accuracy of 98%, recall
#' of 72.9% (or 94.6% depending on metric), precision of 94.6% and ROC value of 0.979.
#' The confusion matrix shows that 10 patients are given an incorrect prediction
#' but this only accounts for 1.8% of the patients.
#' 
#' ### Tuning of Model Hyperparameters
#' Note that the k-Fold Cross Validation incorporated the use of hyperparameters
#' to increase the performance of each model. In the logistic regression and
#' SVM models, the `trainControl()` parameters are further specified to provide
#' additional performance metrics. It is also specified in all models, the method
#' and number of folds that are to be used. If a k-fold value of 5 was used, there
#' would be a lower performance.
#' 
#' It is also important to note that the `caret` package provides automatic tuning.
#' The k parameter in k-NN and C parameter in SVM are adjusted by specifying the
#' appropriate method name. Therefore, while not many parameters are listed, there
#' are many iterations occurring internally for their corresponding hyperparameters.
#' 
#' ### Comparison of Models and Interpretation
#' An ANOVA (Analysis of Variance) test is used to determine the difference in
#' means among different models. The residuals of each model are calculated
#' and combined to conduct an ANOVA test. In summary, the ANOVA results suggest
#' that at least one of the models has a significantly different mean compared to
#' the others. Therefore a Tukey's Honestly Significant Difference (HSD) test is
#' done to determine which models is significantly different from the rest.
#' 
## ----ANOVA-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Residuals of logistic regression model
residuals_log <- as.numeric(test_data$deadstat == "Dead") - 
  as.numeric(predicted_log == "Dead")
# Residuals of k-NN model
residuals_knn <- as.numeric(test_data$deadstat == "Dead") - 
  as.numeric(knn_model == "Dead")
# Residuals of SVM model
residuals_svm <- as.numeric(test_data$deadstat == "Dead") - 
  as.numeric(svm_predictions == "Dead")

# Combine residuals into a data frame
residuals_df <- data.frame(
  Model = rep(c("Logistic Regression", "k-NN", "Support Vector Machine"), 
              each = nrow(test_data)),
  Residuals = c(residuals_log, residuals_knn, residuals_svm)
)

# Fit a linear model and conduct ANOVA
anova_model <- aov(Residuals ~ Model, data = residuals_df)
summary(anova_model)

# Perform Tukey's HSD test
tukey_results <- TukeyHSD(anova_model)
tukey_results

#' 
#' Based on the ANOVA test, the p-value (Pr(>F)) of 0.00288 is less than 0.05, the 
#' significance level. Therefore, there is evidence to reject the null hypothesis
#' that the means of the models are equal. The F-statistic of 5.96 indicates that
#' there are statistically significant differences between the means of the models.
#' The question is which model is statistically different. 
#' 
#' The p-values of the HSD test also further suggests if there is a significant
#' difference between the means of two models. Between the Logistic Regression and
#' k-NN model, the adjusted p-value is 0.0061, which is less than 0.05. It is 
#' suggested that there is a significant difference between the means of Logistic
#' Regression and k-NN. Between the SVM and k-NN, the adjusted p-value is 0.0120, 
#' which is less than 0.05. Again, this suggests that there is a significant
#' difference between the means of Support Vector Machine and k-NN. And finally,
#' between the SVM and Logistic Regression, the adjusted p-value is 0.9734, which
#' is greater than 0.05. This concludes there is no significant difference between
#' the means of Support Vector Machine and Logistic Regression.
#' 
#' Taking all previously reported accuracy measures into account, the Logistic 
#' Regression k-Fold Cross Validation performed best. This is followed with the SVM
#' k-Fold Cross Validation that came in close second. It is only alarming that the
#' k-NN via the holdout method performed poorly. 
#' 
#' ## Deployment
#' 
#' ### Use of Bagging with Homogenous Learners
#' Bagging generates a number of training data sets through bootstrap sampling the
#' original training data. These data sets are then used to create a set of models
#' using the learned algorithm. In the case of classification models, the predictions
#' are then combined using voting to reduce overfitting.
#' 
## ----improvement_bagging_log-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Set seed for reproducibility
set.seed(123)

# Apply bagging to logistic regression
logistic_bagging <- bagging(deadstat ~ ., data = lung, 
                            nbagg = 25, fit = logistic_model)

# Make predictions on the entire data set
predictions_bagging_logistic <- predict(logistic_bagging, newdata = lung)

# Evaluate the performance of bagged logistic regression
conf_matrix_bagging_logistic <- confusionMatrix(predictions_bagging_logistic, 
                                                lung$deadstat)
conf_matrix_bagging_logistic

#' 
#' Above, a 99.8% accuracy is achieved, 100% recall, and 98.9% precision showing a
#' very high performing model. In practical terms, the confusion matrix shows only
#' one instance of a patient given a wrong prediction. In other words, 0.18% of the
#' predictions were wrong. Again, this shows to be a well predictive model. 
#' 
## ----improvement_bagging_knn-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Set seed for reproducibility
set.seed(123)

# Apply bagging to k-NN model
knn_bagging <- bagging(deadstat ~ ., data = lung, 
                            nbagg = 25, fit = knn_model)

# Make predictions on the entire data set
predictions_bagging_knn <- predict(knn_bagging, newdata = lung)

# Evaluate the performance of bagged k-NN model
conf_matrix_bagging_knn <- confusionMatrix(predictions_bagging_knn, 
                                                lung$deadstat)
conf_matrix_bagging_knn

#' 
#' The exact same performance metrics are achieved with the k-NN bagging as with
#' the Logistic Regression bagging, showing another great performing model.
#' 
## ----improvement_bagging_svm-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Set seed for reproducibility
set.seed(123)

# Apply bagging to k-NN model
svm_bagging <- bagging(deadstat ~ ., data = lung, 
                            nbagg = 25, fit = svm_model)

# Make predictions on the entire data set
predictions_bagging_svm <- predict(svm_bagging, newdata = lung)

# Evaluate the performance of bagged k-NN model
conf_matrix_bagging_svm <- confusionMatrix(predictions_bagging_svm, 
                                                lung$deadstat)
conf_matrix_bagging_svm

#' 
#' Interestingly enough, the same high performance metrics are achieved for all 
#' three logistic, k-NN, and SVM models. Note that bagging is best for models
#' that are sensitive to changes in training data. In theory, this would include
#' the SVM model. 
#' 
#' ### Construction of Ensemble Model as a Function
#' A function is now created to provide a stacked ensemble with logistic regression,
#' k-NN, and SVM as base models and logistic regression as the top layer model.
#' The process behind this function is found at:
#' https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/ 
#' 
## ----ensemble--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to construct a stacked ensemble
ensembleFunc <- function(train_data, test_data, outcome_name) {
  # Step 1: Train individual base layer models
  fitControl <- trainControl(
    method = "cv",
    number = 10,
    savePredictions = 'final',
    classProbs = TRUE
  )

  predictors <- names(train_data)[!names(train_data) %in% outcome_name]

  model_lr <- train(train_data[, predictors], train_data[, outcome_name], 
                    method = 'glm', trControl = fitControl, tuneLength = 3,
                    maxit = 1000)
  model_knn <- train(train_data[, predictors], train_data[, outcome_name], 
                     method = 'knn', trControl = fitControl, tuneLength = 3)
  model_svm <- train(train_data[, predictors], train_data[, outcome_name], 
                     method = 'svmLinear', trControl = fitControl, tuneLength = 3)

  # Step 2: Predict using each base layer model for training data and test data
  train_data$OOF_pred_lr <- model_lr$pred$Y[order(model_lr$pred$rowIndex)]
  train_data$OOF_pred_knn <- model_knn$pred$Y[order(model_knn$pred$rowIndex)]
  train_data$OOF_pred_svm <- model_svm$pred$Y[order(model_svm$pred$rowIndex)]

  test_data$OOF_pred_lr <- predict(model_lr, test_data[predictors], 
                                   type = 'prob')$Y
  test_data$OOF_pred_knn <- predict(model_knn, test_data[predictors], 
                                    type = 'prob')$Y
  test_data$OOF_pred_svm <- predict(model_svm, test_data[predictors], 
                                    type = 'prob')$Y

  # Step 3: Train the top layer model (logistic regression) on bottom
  predictors_top <- c('OOF_pred_lr', 'OOF_pred_knn', 'OOF_pred_svm')

  top_layer_model_fit <- train(train_data[, predictors_top],
                               train_data[, outcome_name], method = 'glm', 
                               trControl = fitControl, tuneLength = 3, 
                               maxit = 1000)

  # Step 4: Predict using the top layer with the predictions of bottom layer 
  test_data$stacked_predictions <- predict(top_layer_model_fit, 
                                           test_data[, predictors_top], 
                                           type = 'prob')$Y

  return(test_data$stacked_predictions)
}

#' 
#' 
