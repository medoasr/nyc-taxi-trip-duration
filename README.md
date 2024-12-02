# nyc-taxi-trip-duration
This project predicts the total ride duration of NYC taxi trips using data provided by the NYC Taxi and Limousine Commission. The goal is to build a robust machine learning model that leverages temporal, geographical, and categorical features to estimate trip duration.

Table of Contents
-Introduction
-Dataset
-Exploratory Data Analysis (EDA)
-Feature Engineering
-Modeling
-Results
-Future Work
-How to Use
-Introduction

This project is inspired by the NYC Taxi Duration Prediction competition on Kaggle. The dataset contains information about NYC taxi trips, including:

Pickup time and location
Drop-off location
Number of passengers
Trip duration (target variable)
The dataset is preprocessed and analyzed to create features that improve model performance.

##Dataset
The primary dataset includes:

Numerical Features: e.g., trip distance, number of passengers
Categorical Features: e.g., vendor ID
Geographical Data: Pickup and drop-off coordinates
Temporal Data: Pickup date, day of the week, time of day
Outliers and missing values were carefully handled to ensure data quality.

Exploratory Data Analysis (EDA)
Key insights include:

Trip Duration Distribution: Most trips last between 2.2 to 16.6 minutes. Outliers, such as trips lasting 40+ days, were removed.
Geographical Patterns: Most trips occur within a range of 1–25 km, at speeds of 1–40 km/h.
Temporal Patterns: Longer trip durations occur during afternoon rush hours and in summer months (April, May, June).
Feature Engineering
Log Transformation: Applied to right-skewed features to normalize distributions.
Scaling: Numerical features were standardized using StandardScaler.
One-Hot Encoding: For categorical variables like vendor ID.
Polynomial Features: Higher-order terms (degree=6) were included to capture non-linear relationships.
Modeling
The machine learning pipeline includes:

Data Preprocessing: Log transformation, scaling, and encoding.
Algorithm: A regression model optimized for the task.
Evaluation:
Train R²: 0.7255
Validation R²: 0.675
Outlier removal using z-scores and feature selection significantly improved the model's performance.

Results
The model performs well, with strong correlations identified between trip distance and duration. However, some variance remains unexplained, suggesting room for improvement.

Future Work
Implement version control for datasets and models to improve traceability and error analysis.
Explore advanced models (e.g., Gradient Boosting, Neural Networks) for better accuracy.
Optimize hyperparameters to further improve predictions.
How to Use
Clone this repository:
bash
Copy code
git clone https://github.com/medoasr/nyc-taxi-trip-duration.git
cd nyc-taxi-trip-duration
