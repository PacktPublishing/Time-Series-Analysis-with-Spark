# Databricks notebook source
# MAGIC %md
# MAGIC ## Scaling - LightGBM

# COMMAND ----------

# Install the lightgbm package for gradient boosting framework
# Install the SeqMetrics package for sequence metric calculations
# Install the easy_mpl package for simplified matplotlib plotting
%pip install lightgbm SeqMetrics easy_mpl
%pip install sdv
%pip install optuna optuna-integration

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Loading the dataset

# COMMAND ----------

from pyspark import SparkFiles
import urllib

DATASET_FILE = "ts-spark_ch7_ds1_25mb.csv"
DATASET_URL = f"https://github.com/PacktPublishing/Time-Series-Analysis-with-Spark/raw/main/ch7/{DATASET_FILE}"

# Load the main dataset from a specified path with headers, custom date format, and delimiter
print(f"Ingesting from: {DATASET_URL}")

# option 1 - using sparkContext
#spark.sparkContext.addFile(DATASET_URL)
#df_main = spark.read.csv("file:///" + SparkFiles.get(DATASET_FILE), header=True, dateFormat='yyyy-MM-dd', sep=";")
# option 2 - using urllib
urllib.request.urlretrieve(DATASET_URL, f"/tmp/{DATASET_FILE}")
dbutils.fs.cp(f"file:/tmp/{DATASET_FILE}", f"dbfs:/tmp/{DATASET_FILE}")
df_main = spark.read.csv(f"dbfs:/tmp/{DATASET_FILE}", header=True, dateFormat='yyyy-MM-dd', sep=";")
#

df_main.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Scaled-up dataset (book section)

# COMMAND ----------

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# Convert Spark DataFrame to Pandas DataFrame for compatibility with SDV
pdf_main = df_main.toPandas()

# Initialize metadata object for the dataset
metadata = SingleTableMetadata()

# Automatically detect and set the metadata from the Pandas DataFrame
metadata.detect_from_dataframe(pdf_main)

# Initialize the Gaussian Copula Synthesizer with the dataset metadata
synthesizer = GaussianCopulaSynthesizer(metadata)

# Fit the synthesizer model to the Pandas DataFrame
synthesizer.fit(pdf_main)

# COMMAND ----------

from pyspark.sql import functions as F

num_customers = 5  # Define the number of customer datasets to generate
sample_size = df_main.count()  # Count the number of rows in the original dataset

i = 1
df_all = df_main.withColumn('cust_id', F.lit(i))  # Add a 'cust_id' column to the original dataset with a constant value of 1
for i in range(i+1, num_customers+1):  # Loop to generate synthetic data for additional customers
    synthetic_data = spark.createDataFrame(synthesizer.sample(num_rows=sample_size))  # Generate synthetic data matching the original dataset's size
    synthetic_data = synthetic_data.withColumn('cust_id', F.lit(i))  # Add a 'cust_id' column to the synthetic data with a unique value for each customer
    df_all = df_all.union(synthetic_data)  # Append the synthetic data to the aggregated DataFrame

df_all.cache()  # Cache the resulting DataFrame to optimize subsequent actions

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Column transformations (book section)

# COMMAND ----------

from pyspark.sql import functions as F

# Combine 'Date' and 'Time' into a single 'Date' column of timestamp type
df_all = df_all.withColumn(
    'Date',
    F.to_timestamp(F.concat_ws(' ', F.col('Date'), F.col('Time')), 'd/M/yyyy HH:mm:ss')
)

# Drop the now redundant 'Time' column
df_all = df_all.drop('Time')

# Select only the 'cust_id', 'Date' and 'Global_active_power' columns
df_all = df_all.select('cust_id', 'Date', 'Global_active_power')

# Replace '?' with None and convert 'Global_active_power' to float
df_all = df_all.withColumn(
    'Global_active_power',
    F.when(F.col('Global_active_power') == '?', None)
    .otherwise(F.regexp_replace('Global_active_power', ',', '.').cast('float'))
)

# Sort the DataFrame based on 'cust_id' and 'Date'
df_all = df_all.orderBy('cust_id', 'Date')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Resampling (book section)

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Convert the 'Date' column to a string representing the start of the hour for each timestamp
data_hr = df_all.withColumn('Date', F.date_format('Date', 'yyyy-MM-dd HH:00:00'))

# Group the data by 'cust_id' and the hourly 'Date', then calculate the mean 'Global_active_power' for each group
data_hr = data_hr.groupBy('cust_id', 'Date').agg(F.mean('Global_active_power').alias('Global_active_power'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Calculating lag values (book section)

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql import functions as F

# Define a window specification partitioned by 'cust_id' and ordered by the 'Date' column
windowSpec = Window.partitionBy("cust_id").orderBy("Date")

# Add lagged features to the DataFrame to incorporate past values as features for forecasting
# Apply the lag function to create the lagged column, separately for each 'cust_id'
# Lag by 1, 2, 3, 4, 5, 12, 24, 168 hours (24 hours * 7 days)
lags = [1, 2, 3, 4, 5, 12, 24, 24*7]
for l in lags:
    data_hr = data_hr.withColumn('Global_active_power_lag' + str(l), F.lag(F.col('Global_active_power'), l).over(windowSpec))

# Remove rows with NaN values that were introduced by shifting (lagging) operations
data_hr = data_hr.dropna()
data_hr.cache()

# COMMAND ----------

# Split the data into training and testing sets
# The last 48 observations are used for testing, the rest for training
train_pdf = data_hr.filter('cust_id == 1').toPandas()
train = train_pdf[:-48]
test = train_pdf[-48:]

# COMMAND ----------

# Define the feature set for training based on lagged values of global active power
X_train = train[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']] 
# Define the target variable for the training set
y_train = train['Global_active_power']

X_test = test[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']]
y_test = test['Global_active_power']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 1 - Optuna, LightGBM

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Hyperparameter tuning (book section)

# COMMAND ----------

import joblib
import optuna

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import optuna

def objective(trial):
    # Define the hyperparameter configuration space
    params = {
        "objective": "regression",  # Specify the learning task and the corresponding learning objective
        "metric": "rmse",  # Evaluation metric for the model performance
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),  # Number of boosted trees to fit
         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),  # Learning rate for gradient descent
        "num_leaves": trial.suggest_int("num_leaves", 30, 100),  # Maximum tree leaves for base learners
    }

    model = lgb.LGBMRegressor(**params)  # Initialize the LightGBM model with the trial's parameters
    model.fit(X_train, y_train)  # Train the model with the training dataset
    y_pred = model.predict(X_test)  # Generate predictions for the test dataset
    mape = mean_absolute_percentage_error(y_test, y_pred)  # Calculate the Mean Absolute Percentage Error (MAPE) for model evaluation
    return mape  # Return the MAPE as the objective to minimize

# COMMAND ----------

# Initialize an Optuna study object for hyperparameter optimization, aiming to minimize the objective function
study = optuna.create_study(direction='minimize')
# Execute the optimization process by calling the objective function with the study object, for a specified number of trials
study.optimize(objective, n_trials=10)

# COMMAND ----------

# Retrieve the best trial from the completed study
trial = study.best_trial

# Print the best trial's objective function value (MAPE in this case)
print(f"Best trial accuracy: {trial.value}")
# Print the hyperparameters of the best trial
print("Best trial params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 2 - Optuna with joblib (distributed, 10 trials), LightGBM

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Hyperparameter tuning (book section)

# COMMAND ----------

from joblibspark import register_spark

register_spark() # This line registers Apache Spark as the backend for parallel computing with Joblib, enabling distributed computing capabilities for Joblib-based parallel tasks.

# COMMAND ----------

# Create a new study object with the goal of minimizing the objective function
study2 = optuna.create_study(direction='minimize')
# Set Apache Spark as the backend for parallel execution of trials with unlimited jobs
with joblib.parallel_backend("spark", n_jobs=-1):
    # Optimize the study by evaluating the objective function over 10 trials
    study2.optimize(objective, n_trials=10)

# COMMAND ----------

# Retrieve the best trial from the optimization study
trial = study2.best_trial

# Print the best trial's objective function value, typically accuracy or loss
print(f"Best trial accuracy: {trial.value}")
print("Best trial params: ")

# Iterate through the best trial's hyperparameters and print them
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# COMMAND ----------

# Initialize the LightGBM Regressor model with the best hyperparameters found from the optimization study
final_model = lgb.LGBMRegressor(boosting_type='gbdt', **trial.params)
# Fit the model on the training data
final_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test dataset
y_pred = final_model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Calculate Mean Squared Error (MSE) between the actual and predicted values
mse = mean_squared_error(y_test, y_pred)
# Calculate Mean Absolute Error (MAE) between the actual and predicted values
mae = mean_absolute_error(y_test, y_pred)
# Calculate Mean Absolute Percentage Error (MAPE) between the actual and predicted values
mape = mean_absolute_percentage_error(y_test, y_pred)

# Print the evaluation metrics to assess model performance
print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test MAPE: {mape}")

# COMMAND ----------

import matplotlib.pyplot as plt

# Initialize a new figure with specified dimensions
plt.figure(figsize=(10, 6))
# Plot the last 150 data points from the training set for visual comparison
plt.plot(train[-150:].index, train[-150:]['Global_active_power'], label='Train')
# Plot all data points from the test set
plt.plot(test.index, test['Global_active_power'], label='Test')
# Overlay the predicted values on the test set for comparison
plt.plot(test.index, y_pred, label='Forecast')
# Add a label to the x-axis
plt.xlabel('Date')
# Add a label to the y-axis
plt.ylabel('Global_active_power')
# Set a title for the plot
plt.title('LightGBM Forecast vs Actuals')
# Show a legend to label each line plot
plt.legend()
# Render the plot to the screen
plt.show()

# Start a new figure for plotting test set and forecast only
plt.figure(figsize=(10, 6))
# Plot all data points from the test set
plt.plot(test.index, test['Global_active_power'], label='Test')
# Overlay the predicted values on the test set for comparison
plt.plot(test.index, y_pred, label='Forecast')
# Add a label to the x-axis
plt.xlabel('Date')
# Add a label to the y-axis
plt.ylabel('Global_active_power')
# Set a title for the plot
plt.title('LightGBM Forecast vs Actuals')
# Show a legend to label each line plot
plt.legend()
# Render the plot to the screen
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 3 - Optuna with joblib (distributed, 30 trials), LightGBM

# COMMAND ----------

study3 = optuna.create_study(direction='minimize')
with joblib.parallel_backend("spark", n_jobs=-1):
    study3.optimize(objective, n_trials=30)

# COMMAND ----------

trial = study3.best_trial

print(f"Best trial accuracy: {trial.value}")
print("Best trial params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# COMMAND ----------

final_model = lgb.LGBMRegressor(**trial.params)
final_model.fit(X_train, y_train)

# Predict on the test set
y_pred = final_model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Evaluate the model using MSE, MAE, and MAPE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test MAPE: {mape}")

# COMMAND ----------

import matplotlib.pyplot as plt

# Set the size of the plot
plt.figure(figsize=(10, 6))
# Plot the last 150 points of the training set
plt.plot(train[-150:].index, train[-150:]['Global_active_power'], label='Train')
# Plot the entire test set
plt.plot(test.index, test['Global_active_power'], label='Test')
# Plot the predicted values for the test set
plt.plot(test.index, y_pred, label='Forecast')
# Label the x-axis as 'Date'
plt.xlabel('Date')
# Label the y-axis as 'Global_active_power'
plt.ylabel('Global_active_power')
# Set the title of the plot
plt.title('LightGBM Forecast vs Actuals')
# Display the legend to differentiate between the plotted lines
plt.legend()
# Display the plot
plt.show()

# Set the size of the plot
plt.figure(figsize=(10, 6))
# Plot the entire test set
plt.plot(test.index, test['Global_active_power'], label='Test')
# Plot the predicted values for the test set
plt.plot(test.index, y_pred, label='Forecast')
# Label the x-axis as 'Date'
plt.xlabel('Date')
# Label the y-axis as 'Global_active_power'
plt.ylabel('Global_active_power')
# Set the title of the plot
plt.title('LightGBM Forecast vs Actuals')
# Display the legend to differentiate between the plotted lines
plt.legend()
# Display the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 4 - Pandas, TimeSeriesSplit, GridSearchCV, LightGBM

# COMMAND ----------

# Split the data into training and testing sets
# The last 48 observations are used for testing, the rest for training
train_pdf = data_hr.filter('cust_id == 1').toPandas()
train = train_pdf[:-48]
test = train_pdf[-48:]

# COMMAND ----------

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Extract features and target variable for training and testing sets
X_train = train[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']]
y_train = train['Global_active_power']
X_test = test[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']]
y_test = test['Global_active_power']

# Define the parameter grid for LightGBM
param_grid = {
    'num_leaves': [30, 50, 100],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200]
}

# Initialize LightGBM regressor
lgbm = lgb.LGBMRegressor()

# Setup TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=10)

# Configure and run GridSearchCV
gsearch = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=tscv)
gsearch.fit(X_train, y_train)

# Output the best parameters from Grid Search
print(f"Best Parameters: {gsearch.best_params_}")

# Train the model with the best parameters found
best_params = gsearch.best_params_
final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train)

# Predict on the test set
y_pred = final_model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Evaluate the model using MSE, MAE, and MAPE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test MAPE: {mape}")

# COMMAND ----------

from SeqMetrics import RegressionMetrics, plot_metrics

# Initialize the RegressionMetrics object with actual and predicted values
er = RegressionMetrics(y_test, y_pred)

# Calculate all available regression metrics
metrics = er.calculate_all()

# Plot the calculated metrics using a color gradient of "Blues"
plot_metrics(metrics, color="Blues")
# Print the Symmetric Mean Absolute Percentage Error (SMAPE)
print(f"Test SMAPE: {metrics['smape']}")
# Print the Weighted Absolute Percentage Error (WAPE)
print(f"Test WAPE: {metrics['wape']}")

# COMMAND ----------

import matplotlib.pyplot as plt

# Set the size of the plot
plt.figure(figsize=(10, 6))
# Plot the last 150 points of the training set
plt.plot(train[-150:].index, train[-150:]['Global_active_power'], label='Train')
# Plot the entire test set
plt.plot(test.index, test['Global_active_power'], label='Test')
# Plot the predicted values for the test set
plt.plot(test.index, y_pred, label='Forecast')
# Label the x-axis as 'Date'
plt.xlabel('Date')
# Label the y-axis as 'Global_active_power'
plt.ylabel('Global_active_power')
# Set the title of the plot
plt.title('LightGBM Forecast vs Actuals')
# Display the legend to differentiate between the plotted lines
plt.legend()
# Display the plot
plt.show()

# Set the size of the plot
plt.figure(figsize=(10, 6))
# Plot the entire test set
plt.plot(test.index, test['Global_active_power'], label='Test')
# Plot the predicted values for the test set
plt.plot(test.index, y_pred, label='Forecast')
# Label the x-axis as 'Date'
plt.xlabel('Date')
# Label the y-axis as 'Global_active_power'
plt.ylabel('Global_active_power')
# Set the title of the plot
plt.title('LightGBM Forecast vs Actuals')
# Display the legend to differentiate between the plotted lines
plt.legend()
# Display the plot
plt.show()

# COMMAND ----------

import shap

# Initialize a SHAP TreeExplainer with the trained model
explainer = shap.TreeExplainer(final_model)

# Select features for SHAP analysis
X = data_hr[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']].toPandas()

# Compute SHAP values for the selected features
shap_values = explainer(X)

# Generate and display a summary plot of the SHAP values
shap.summary_plot(shap_values, X)

# COMMAND ----------

# Plot a SHAP waterfall plot for the first observation's SHAP values to visualize the contribution of each feature
shap.plots.waterfall(shap_values[0])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 5 - Pandas, TimeSeriesSplit, GridSearchCV, LightGBM

# COMMAND ----------

# Split the data into training and testing sets
# The last 48 observations are used for testing, the rest for 
train_pdf = data_hr.filter('cust_id == 1').toPandas()
train = train_pdf[:-48]
test = train_pdf[-48:]

# COMMAND ----------

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Extract features and target variable for training and testing sets
X_train = train[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']]
y_train = train['Global_active_power']
X_test = test[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']]
y_test = test['Global_active_power']

# Define the parameter grid for LightGBM
param_grid = {
    'num_leaves': [30, 50, 100],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200]
}

# Initialize LightGBM regressor
lgbm = lgb.LGBMRegressor()

# Setup TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=10)

# Configure and run GridSearchCV
gsearch = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=tscv)
gsearch.fit(X_train, y_train)

# Output the best parameters from Grid Search
print(f"Best Parameters: {gsearch.best_params_}")

# Train the model with the best parameters found
best_params = gsearch.best_params_
final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train)

# Predict on the test set
y_pred = final_model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Evaluate the model using MSE, MAE, and MAPE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test MAPE: {mape}")

# COMMAND ----------

from SeqMetrics import RegressionMetrics, plot_metrics

# Initialize the RegressionMetrics object with actual and predicted values
er = RegressionMetrics(y_test, y_pred)

# Calculate all available regression metrics
metrics = er.calculate_all()

# Plot the calculated metrics using a color gradient of "Blues"
plot_metrics(metrics, color="Blues")
# Print the Symmetric Mean Absolute Percentage Error (SMAPE)
print(f"Test SMAPE: {metrics['smape']}")
# Print the Weighted Absolute Percentage Error (WAPE)
print(f"Test WAPE: {metrics['wape']}")

# COMMAND ----------

import matplotlib.pyplot as plt

# Set the size of the plot
plt.figure(figsize=(10, 6))
# Plot the last 150 points of the training set
plt.plot(train[-150:].index, train[-150:]['Global_active_power'], label='Train')
# Plot the entire test set
plt.plot(test.index, test['Global_active_power'], label='Test')
# Plot the predicted values for the test set
plt.plot(test.index, y_pred, label='Forecast')
# Label the x-axis as 'Date'
plt.xlabel('Date')
# Label the y-axis as 'Global_active_power'
plt.ylabel('Global_active_power')
# Set the title of the plot
plt.title('LightGBM Forecast vs Actuals')
# Display the legend to differentiate between the plotted lines
plt.legend()
# Display the plot
plt.show()

# Set the size of the plot
plt.figure(figsize=(10, 6))
# Plot the entire test set
plt.plot(test.index, test['Global_active_power'], label='Test')
# Plot the predicted values for the test set
plt.plot(test.index, y_pred, label='Forecast')
# Label the x-axis as 'Date'
plt.xlabel('Date')
# Label the y-axis as 'Global_active_power'
plt.ylabel('Global_active_power')
# Set the title of the plot
plt.title('LightGBM Forecast vs Actuals')
# Display the legend to differentiate between the plotted lines
plt.legend()
# Display the plot
plt.show()


# COMMAND ----------

import shap

# Initialize a SHAP TreeExplainer with the trained model
explainer = shap.TreeExplainer(final_model)

# Select features for SHAP analysis
X = data_hr[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']].toPandas()

# Compute SHAP values for the selected features
shap_values = explainer(X)

# Generate and display a summary plot of the SHAP values
shap.summary_plot(shap_values, X)

# COMMAND ----------

# Plot a SHAP waterfall plot for the first observation's SHAP values to visualize the contribution of each feature
shap.plots.waterfall(shap_values[0])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 6 - VectorAssembler, Pipeline, XGBoost

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Single model in parallel (book section)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Define a list to hold the names of the lag feature columns
inputCols = []
# Loop through the list of lag intervals to create feature column names
for l in lags:
    inputCols.append('Global_active_power_lag' + str(l))

# Initialize VectorAssembler with the created feature column names and specify the output column name
assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
# Transform the data to assemble the features into a vector (commented out as it's an example of usage)
#data_hr_vect = assembler.transform(data_hr).select("cust_id", "Date", "features", "Global_active_power")

# Example of how to sort the transformed data by customer ID and date (commented out as it's an example of usage)
#data_hr_vect = data_hr_vect.orderBy("cust_id", "Date")

# COMMAND ----------

from xgboost.spark import SparkXGBRegressor

# Initialize the SparkXGBRegressor for the regression task. 
# `num_workers` is set to the default parallelism level of the Spark context to utilize all available cores.
# `label_col` specifies the target variable column name for prediction.
# `missing` is set to 0.0 to handle missing values in the dataset.
xgb_model = SparkXGBRegressor(num_workers=sc.defaultParallelism, label_col="Global_active_power", missing=0.0)

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize the parameter grid for hyperparameter tuning
# - max_depth: specifies the maximum depth of the trees in the model
# - n_estimators: defines the number of trees in the ensemble
paramGrid = ParamGridBuilder()\
  .addGrid(xgb_model.max_depth, [5, 10])\
  .addGrid(xgb_model.n_estimators, [30, 100])\
  .build()

# Initialize the regression evaluator for model evaluation
# - metricName: specifies the metric to use for evaluation, here RMSE (Root Mean Squared Error)
# - labelCol: the name of the label column
# - predictionCol: the name of the prediction column
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol=xgb_model.getLabelCol(),
                                predictionCol=xgb_model.getPredictionCol())

# Initialize the CrossValidator for hyperparameter tuning
# - estimator: the model to be tuned
# - evaluator: the evaluator to be used for model evaluation
# - estimatorParamMaps: the grid of parameters to be used for tuning
cv = CrossValidator(estimator=xgb_model, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

from pyspark.ml import Pipeline

# Initialize a Pipeline object with two stages: a feature assembler and a cross-validator for model tuning
pipeline = Pipeline(stages=[assembler, cv])

# COMMAND ----------

# Filter the dataset for customer with cust_id equal to 1
train_hr = data_hr.filter('cust_id == 1')

# Create a Spark DataFrame excluding the last 48 records for training
train_hr = spark.createDataFrame(train_hr.head(train_hr.count() - 48))

# Fit the pipeline model to the training data
pipelineModel = pipeline.fit(train_hr)

# COMMAND ----------

# Filter the dataset for customer with cust_id equal to 1 for testing
test_hr = data_hr.filter('cust_id == 1')
# Create a Spark DataFrame including the last 48 records for testing
test_hr = spark.createDataFrame(train_hr.tail(48))
# The following lines are commented out as they are not used in this snippet
#X_test = test_hr.select('Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168')
#y_test = test_hr.select('Global_active_power')
# Apply the trained pipeline model to the test data to generate predictions
predictions = pipelineModel.transform(test_hr)

# COMMAND ----------

# Evaluate the model's performance using the Root Mean Squared Error (RMSE) metric
rmse = evaluator.evaluate(predictions)
# Print the calculated RMSE to assess the model's prediction accuracy
print("Test RMSE: %g" % rmse)

# COMMAND ----------

# Convert the 'Global_active_power' column from the test DataFrame to a Pandas DataFrame for evaluation
y_test = test_hr.select('Global_active_power').toPandas()
# Convert the 'prediction' column from the predictions DataFrame to a Pandas DataFrame for evaluation
y_pred = predictions.select('prediction').toPandas()

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Calculate Mean Squared Error (MSE) between actual and predicted values
mse = mean_squared_error(y_test, y_pred)
# Calculate Mean Absolute Error (MAE) between actual and predicted values
mae = mean_absolute_error(y_test, y_pred)
# Calculate Mean Absolute Percentage Error (MAPE) between actual and predicted values
mape = mean_absolute_percentage_error(y_test, y_pred)
# Display the calculated MSE, MAE, and MAPE to evaluate model performance
print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test MAPE: {mape}")

# COMMAND ----------

from SeqMetrics import RegressionMetrics, plot_metrics

# Initialize the RegressionMetrics object with actual (y_test) and predicted (y_pred) values
er = RegressionMetrics(y_test, y_pred)

# Calculate all available regression metrics and store them in the variable 'metrics'
metrics = er.calculate_all()

# Plot the calculated metrics using a color gradient of "Blues"
plot_metrics(metrics, color="Blues")

# Print the Symmetric Mean Absolute Percentage Error (SMAPE) from the calculated metrics
print(f"Test SMAPE: {metrics['smape']}")

# Print the Weighted Absolute Percentage Error (WAPE) from the calculated metrics
print(f"Test WAPE: {metrics['wape']}")

# COMMAND ----------

import matplotlib.pyplot as plt

# Initialize a figure with a specified size for plotting training and test data
plt.figure(figsize=(10, 6))
# Plot the last 150 data points from the training set to visualize recent trends
plt.plot(train[-150:].index, train[-150:]['Global_active_power'], label='Train')
# Plot the entire test dataset to compare against the forecast
plt.plot(test.index, test['Global_active_power'], label='Test')
# Overlay the forecasted values on the test dataset for direct comparison
plt.plot(test.index, y_pred, label='Forecast')
# Add a label to the x-axis indicating the data represents dates
plt.xlabel('Date')
# Add a label to the y-axis indicating the data represents global active power consumption
plt.ylabel('Global_active_power')
# Set a title for the plot to indicate it shows a forecast comparison
plt.title('LightGBM Forecast vs Actuals')
# Add a legend to the plot to identify the train, test, and forecast lines
plt.legend()
# Render the plot to display it
plt.show()

# Initialize a second figure with a specified size for plotting test data and forecast
plt.figure(figsize=(10, 6))
# Plot the entire test dataset to show actual consumption
plt.plot(test.index, test['Global_active_power'], label='Test')
# Overlay the forecasted values on the test dataset for evaluation
plt.plot(test.index, y_pred, label='Forecast')
# Add a label to the x-axis indicating the data represents dates
plt.xlabel('Date')
# Add a label to the y-axis indicating the data represents global active power consumption
plt.ylabel('Global_active_power')
# Set a title for the plot to indicate it shows a forecast comparison
plt.title('LightGBM Forecast vs Actuals')
# Add a legend to the plot to identify the test and forecast lines
plt.legend()
# Render the plot to display it
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 7 - applyInPandas, LightGBM

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

train_model_result_schema = StructType([
  StructField("cust_id", IntegerType()),
  StructField("rmse", FloatType()),
  StructField("mape", FloatType())          
])

# COMMAND ----------

import mlflow.sklearn
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
  """
  Trains LGBMRegressor model on a group of data 
  """
  #collect information about the current DataFrame that is being processed
  #get the cust_id for which model is being trained
  cust_id = df_pandas["cust_id"].iloc[0]
  
  # Create the Gradient Boosting Regression model
  #gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=12)
  lgbm = lgb.LGBMRegressor()

  # Define features to train on and the label
  X = df_pandas[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']]
  y = df_pandas['Global_active_power']

  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False, random_state=12)

  # Train the model on the training data
  #gbr.fit(X_train, y_train)
  lgbm.fit(X_train, y_train)

  # Evaluate model
  #y_pred = gbr.predict(X_test)
  y_pred = lgbm.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
  mape = mean_absolute_percentage_error(y_test, y_pred)
  
  return_df = pd.DataFrame([[cust_id, rmse, mape]], 
        columns=["cust_id", "rmse", "mape"])

  return return_df 

# COMMAND ----------

from pyspark.sql.functions import lit
#explicitly set the experiment for the model trainings we are going to perform
train_model_result_df = (data_hr
                          .groupby("cust_id")
                          .applyInPandas(train_model, schema=train_model_result_schema)
                          .cache()
)

display(train_model_result_df.orderBy('cust_id'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 8 - applyInPandas, TimeSeriesSplit, GridSearchCV, LightGBM

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Multiple models in parallel (book section)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

# Define the schema for the DataFrame that will store information about trained models
train_model_result_schema = StructType([
  StructField("cust_id", IntegerType()),  # Customer ID as an integer
  StructField("best_params", StringType()),  # Best parameters found during model tuning, stored as a string
  StructField("rmse", FloatType()),  # Root Mean Squared Error of the model predictions, as a float
  StructField("mape", FloatType())  # Mean Absolute Percentage Error of the model predictions, as a float          
])

# COMMAND ----------

import mlflow.sklearn
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    # Extract the customer ID for which the model is being trained
    cust_id = df_pandas["cust_id"].iloc[0]
    
    # Select features and target variable from the DataFrame
    X = df_pandas[['Global_active_power_lag1', 'Global_active_power_lag2', 'Global_active_power_lag3', 'Global_active_power_lag4', 'Global_active_power_lag5', 'Global_active_power_lag12', 'Global_active_power_lag24', 'Global_active_power_lag168']]
    y = df_pandas['Global_active_power']

    # Split the dataset into training and testing sets, preserving time order
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=12)

    # Define the hyperparameter space for LightGBM model tuning
    param_grid = {
        'num_leaves': [30, 50, 100],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [50, 100, 200]
    }

    # Initialize the LightGBM regressor model
    lgbm = lgb.LGBMRegressor()

    # Initialize TimeSeriesSplit for cross-validation to respect time series data structure
    tscv = TimeSeriesSplit(n_splits=10)

    # Perform grid search with cross-validation
    gsearch = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=tscv)
    gsearch.fit(X_train, y_train)

    # Extract the best hyperparameters
    best_params = gsearch.best_params_

    # Train the final model using the best parameters
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = final_model.predict(X_test)
    # Calculate RMSE and MAPE metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Prepare the results DataFrame to return
    return_df = pd.DataFrame([[cust_id, str(best_params), rmse, mape]], 
                             columns=["cust_id", "best_params", "rmse", "mape"])

    return return_df

# COMMAND ----------

from pyspark.sql.functions import lit

# Group the data by customer ID and apply the train_model function to each group using Pandas UDF
# The schema for the resulting DataFrame is defined by trained_models_info_schema
# Cache the resulting DataFrame to optimize performance for subsequent actions
train_model_result_df = (data_hr
                          .groupby("cust_id")
                          .applyInPandas(train_model, schema=train_model_result_schema)
                          .cache()
)

# Display the model training information DataFrame, ordered by customer ID
display(train_model_result_df.orderBy('cust_id'))