#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, regexp_extract, dayofmonth, month, year, to_date
from pyspark.ml.feature import Imputer
from pyspark.sql.types import IntegerType, StringType, FloatType, DateType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession


# In[ ]:


spark = SparkSession.builder.appName("AirQualityAnalysis").getOrCreate()


# In[ ]:


df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/oadegoke@gmu.edu/Air_Quality_2-3.csv")


# In[ ]:


df1.show(5)


# In[ ]:


df1.printSchema()


# In[ ]:


df1.columns


# In[ ]:


df1.dtypes


# In[ ]:


for col_name in df1.columns:
    missing_count = df1.filter(df1[col_name].isNull()).count()
    if missing_count > 0:
        print(f"Column {col_name} has {missing_count} missing values")


# In[ ]:


df1 = df1.drop('Message')


# In[ ]:


df1 = df1.withColumn("Start_Date", to_date(col("Start_Date"), "MM/dd/yyyy"))


# In[ ]:


print(df1.head(5))


# In[ ]:


df1.select('Time Period').distinct().collect()


# In[ ]:


from pyspark.sql.functions import col, when, lit, split

# Extract season (optional)
df1 = df1.withColumn("Season", 
                   when(col("Time Period").contains("Winter"), split(col("Time Period"), " ")[0])
                   .when(col("Time Period").contains("Summer"), split(col("Time Period"), " ")[0])
                   .otherwise(lit("NA")))  # Replace with placeholder for non-seasonal periods

# Extract year range for seasons (optional)
df1 = df1.withColumn("Year Range", 
                   when(col("Season").isNotNull(), 
                        regexp_extract(col("Time Period"), r"\d{4}-\d{4}$", 0))
                   .otherwise(lit("NA")))


# In[ ]:


# Fill NA with "unknown" and separate annual averages
df1 = df1.withColumn("Time Period", 
                   when(col("Time Period").isNull(), lit("unknown"))  # Replace NA with "unknown"
                   .when(col("Time Period").contains("Annual Average"), 
                         split(col("Time Period"), " ")[1])  # Extract year from "Annual Average"
                   .otherwise(col("Time Period")))


# In[ ]:


print(df1.head(5))


# In[ ]:


df1 = df1.withColumn('Unique ID', col('Unique ID').cast(IntegerType())) \
              .withColumn('Indicator ID', col('Indicator ID').cast(IntegerType())) \
              .withColumn('Geo Join ID', col('Geo Join ID').cast(IntegerType())) \
              .withColumn('Data Value', col('Data Value').cast(IntegerType()))

df_casted = df1
df_casted.printSchema()


# In[ ]:


print(df_casted.head(5))


# In[ ]:





# In[ ]:


# string_cols = ['Name', 'Measure', 'Measure Info', 'Geo Type Name', 'Geo Place Name']
# indexers = [StringIndexer(inputCol=c, outputCol=c+"_index").fit(df_casted) for c in string_cols]

# df_casted = df_casted.withColumn('Day', dayofmonth('Start_Date'))
# df_casted = df_casted.withColumn('Month', month('Start_Date'))
# df_casted = df_casted.withColumn('Year', year('Start_Date'))

# # Define the VectorAssembler with numeric features and the indices of the categorical features
# assembler_inputs = [c + "_index" for c in string_cols] + ['Day', 'Month', 'Year', 'Geo Join ID']
# vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# pipeline = Pipeline(stages=indexers + [vec_assembler])

# df_transformed = pipeline.fit(df_casted).transform(df_casted)

# # Split the data into training and test sets
# train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=42)

# # Define the linear regression model
# lr = LinearRegression(featuresCol="features", labelCol="Data Value")

# # Train the model on the training data
# lr_model = lr.fit(train_data)

# # Make predictions on the test data
# predictions = lr_model.transform(test_data)

# # Show some predictions
# predictions.select("prediction", "Data Value", "features").show(5)


# In[ ]:


df_casted.head()


# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


categorical_cols = ['Name', 'Measure', 'Measure Info', 'Geo Type Name', 'Geo Place Name', 'Season']
for c in categorical_cols:
    df_casted = df_casted.fillna('unknown', subset=[c])

numeric_cols = ['Geo Join ID']  # Add other numeric feature column names here
# Filling null values for numerical columns with 0 (assuming 'numeric_cols' is defined)
df_casted = df_casted.fillna(0, subset=numeric_cols)

# Defining StringIndexer with handleInvalid parameter set to 'keep'
indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep").fit(df_casted) for c in categorical_cols]

# Defining OneHotEncoders for the categorical columns
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=indexer.getOutputCol()+"_vec") for indexer in indexers]

# Assembling all features into a single vector column
assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] + numeric_cols, outputCol="features")

# Defining the RandomForestRegressor model
rf = RandomForestRegressor(featuresCol="features", labelCol="Data Value")

# Building the pipeline with all transformations and the estimator
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# Splitting the data into training and test sets
train_data, test_data = df_casted.randomSplit([0.8, 0.2], seed=42)

# Fitting the model
model = pipeline.fit(train_data)

# Making predictions
predictions = model.transform(test_data)

# Selecting and showing the predictions
predictions.select("prediction", "Data Value", "features").show(5)

# Evaluating the model using RMSE
evaluator = RegressionEvaluator(labelCol="Data Value", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[ ]:


# R^2
evaluator_r2 = RegressionEvaluator(labelCol="Data Value", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print("R-squared (R^2) on test data = %g" % r2)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Get feature importances
importances = model.stages[-1].featureImportances

# Convert to a list with column names
importances_list = [(assembler.getInputCols()[i], importances[i]) for i in range(len(assembler.getInputCols()))]
importances_df = pd.DataFrame(importances_list, columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importances_df["Feature"], importances_df["Importance"])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from RandomForestRegressor')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()


# In[ ]:


# Convert predictions to a Pandas DataFrame
predictions_df = predictions.select("prediction", "Data Value").toPandas()

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(predictions_df["Data Value"], predictions_df["prediction"], alpha=0.5)
plt.xlabel('Actual Data Value')
plt.ylabel('Predicted Data Value')
plt.title('Actual vs Predicted Data Value')
plt.plot([predictions_df["Data Value"].min(), predictions_df["Data Value"].max()],
         [predictions_df["Data Value"].min(), predictions_df["Data Value"].max()], 'k--')
plt.show()


# NOW FOR TEMPPRAL TREND ANALYSIS

# In[ ]:


df2 = df_casted
df4 = df_casted.toPandas()
df4.head()


# In[ ]:


from pyspark.sql.functions import year, month, dayofmonth, dayofweek, quarter

df2 = df2.withColumn('Year', year(col('Start_Date')))
df2 = df2.withColumn('Month', month(col('Start_Date')))
df2 = df2.withColumn('DayOfMonth', dayofmonth(col('Start_Date')))
df2 = df2.withColumn('DayOfWeek', dayofweek(col('Start_Date')))
df2 = df2.withColumn('Quarter', quarter(col('Start_Date')))

# Update numeric_cols to include the new time-based features
numeric_cols += ['Year', 'Month', 'DayOfMonth', 'DayOfWeek', 'Quarter']


# In[ ]:


import statsmodels.api as sm

# Convert Spark DataFrame to Pandas DataFrame for time series analysis
df_pd = df2.toPandas()

# Assuming df_pd is indexed by the date, and you have a column "Data Value" for the time series
mod = sm.tsa.statespace.SARIMAX(df_pd['Data Value'],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

# You can now use results to make predictions and evaluate them.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df_casted is already ordered by date and predictions is your model output
df_plot = df_pd # convert to Pandas DataFrame if not already done
df_plot['prediction'] = predictions.toPandas()['prediction']

plt.figure(figsize=(15, 7))
sns.lineplot(data=df_plot, x='Start_Date', y='Data Value', label='Actual')
sns.lineplot(data=df_plot, x='Start_Date', y='prediction', label='Predicted')
plt.title('Air Quality Over Time')
plt.xlabel('Date')
plt.ylabel('Air Quality Value')
plt.legend()
plt.show()


# In[ ]:


df_plot.head()


# In[ ]:


df_plot['residuals'] = df_plot['Data Value'] - df_plot['prediction']

plt.figure(figsize=(15, 7))
sns.lineplot(data=df_plot, x='Start_Date', y='residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Convert the Spark DataFrame to a Pandas DataFrame for plotting
predictions_pd = predictions.select("prediction", "Data Value", "Start_Date").toPandas()

# Set 'Start_Date' as the index for time series plotting
predictions_pd.set_index('Start_Date', inplace=True)
predictions_pd.sort_index(inplace=True)

# Time Series Plot
plt.figure(figsize=(12, 6))
plt.plot(predictions_pd['Data Value'], label='Actual')
plt.plot(predictions_pd['prediction'], label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Air Quality Over Time')
plt.xlabel('Date')
plt.ylabel('Air Quality Value')
plt.legend()
plt.show()



# In[ ]:


# Calculate residuals
df_plot['Residuals'] = df_plot['Data Value'] - df_plot['prediction']

# Plotting residuals
plt.figure(figsize=(10, 6))
plt.scatter(df_plot['Start_Date'], df_plot['Residuals'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residuals Over Time')
plt.show()


# In[ ]:




