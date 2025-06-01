# Music Genre Classification

This notebook explores various machine learning models to classify music genres based on audio features.

## Dataset

The dataset used in this notebook is the "Music Genre Classification" dataset from Kaggle, which contains audio features and corresponding genre labels for a large collection of songs.

## Notebook Overview

The notebook follows a standard machine learning workflow:

1.  **Data Loading and Initial Exploration**: The raw data is loaded, and basic information about the dataset, including column types, missing values, and descriptive statistics, is examined.
2.  **Data Preprocessing**:
    *   Irrelevant columns ('Artist Name', 'Track Name') are removed.
    *   The 'duration_in min/ms' column is converted to seconds and renamed.
    *   Duplicate records are identified and removed.
3.  **Univariate Analysis**: Histograms and box plots are used to visualize the distribution and identify outliers in each feature.
4.  **Handling Missing Values**: Missing values in 'popularity', 'instrumentalness', and 'key' are imputed using appropriate strategies (mean, median, or a specific value).
5.  **Outlier Removal**: Outliers are handled using the IQR method to clip extreme values.
6.  **Feature Transformation**: Logarithmic transformation is applied to skewed features ('speechiness', 'acousticness', 'liveness', 'instrumentalness', 'duration_in_min_sg') to achieve a more normal distribution.
7.  **Class Distribution Analysis**: The distribution of music genres (classes) is analyzed, revealing class imbalance.
8.  **Feature Engineering**:
    *   Correlation analysis is performed to understand relationships between features and the target variable.
    *   Features with zero variance and high collinearity are removed.
    *   A subset of potentially useful features for prediction is selected.
9.  **Data Splitting**: The dataset is split into training, validation, and test sets while maintaining the original class distribution using stratification.
10. **Handling Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to balance the class distribution.
11. **Feature Scaling**: Features are standardized using StandardScaler to ensure all features have a similar scale.
12. **Model Training and Evaluation**:
    *   **Logistic Regression**: Hyperparameter tuning is performed to select the optimal 'C' value. The model is trained on the balanced and scaled training data and evaluated on the test set using classification reports, confusion matrices, and ROC curves.
    *   **Decision Tree**: Hyperparameter tuning is performed to select optimal parameters. The model is trained and evaluated similarly to Logistic Regression.
    *   **Support Vector Machine (SVC)**: Hyperparameter tuning is performed to select the optimal 'C' value and kernel. The model is trained and evaluated similarly to the other models.
13. **Performance Comparison**: The performance of the three models is compared using classification reports, confusion matrices, and ROC curves to identify the best-performing model for this task.

## Models Evaluated

*   **Logistic Regression**: A linear model for classification.
*   **Decision Tree**: A tree-based model that partitions the data based on features.
*   **Support Vector Machine (SVC)**: A powerful model that finds an optimal hyperplane to separate classes.

## Results

The evaluation results show that **SVC** generally outperforms Logistic Regression and Decision Tree in terms of overall accuracy and F1-scores across most music genres, particularly for classes 0, 3, 4, and 7. Class 7 is consistently well-predicted by all models. However, all models struggle with predicting Class 1.

## Recommendations

*   **Prioritize SVC** for music genre classification based on the current analysis.
*   Further investigate **Class 1** to understand the reasons for poor performance and explore potential solutions like collecting more data for this class or trying different feature engineering techniques.
*   Consider applying **class balancing techniques** or **weight adjustments** to improve the performance of Logistic Regression and Decision Tree for underperforming classes.
