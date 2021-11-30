# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# Common imports
import pandas as pd
import numpy as np
 
# pyspark imports
import pyspark.sql.functions as f
from pyspark.sql.functions import countDistinct
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/HillsboroughCountyData___Updated_Clean.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

print("DF Shape: ",(df.count() , len(df.columns)))
display(df)


# COMMAND ----------

df.select('Acreage', 'SiteZip', 'AssessedValue').describe().show()

# COMMAND ----------

# Create a view or table

temp_table_name = "HillsboroughCountyData___Updated_Clean_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

#dependent variable 
data = df.drop('AssessedValue')

# COMMAND ----------

# Create a 70-30 train test split
train_data,test_data=data.randomSplit([0.7,0.3])

# COMMAND ----------

# Fit training data
fitted_pipe = pipe.fit(train_data)

# COMMAND ----------

train_data = fitted_pipe.transform(train_data)
display(train_data)

# COMMAND ----------

propertytype_indexer = StringIndexer(inputCol='PropertyType',outputCol='propertytype_index',handleInvalid='keep')
siteaddress_indexer = StringIndexer(inputCol='SiteAddress',outputCol='siteadress_index',handleInvalid='keep')
sitecity_indexer = StringIndexer(inputCol='SiteCity',outputCol='sitecity_index',handleInvalid='keep')
homestead = StringIndexer(inputCol='Homestead',outputCol='Homestead_index',handleInvalid='keep')
neighborhood_indexer = StringIndexer(inputCol='Neighborhood',outputCol='Neighborhood_index',handleInvalid='keep')
lastsaledate_indexer = StringIndexer(inputCol='LastSaleDate',outputCol='LastSaleDate_index',handleInvalid='keep')


# COMMAND ----------


# OneHotEncoderEstimator converts the indexed data into a vector which will be effectively handled by Logistic Regression model
 
data_encoder = OneHotEncoder(inputCols=['propertytype_index', 'siteadress_index', 'sitecity_index', 'Homestead_index', 'Neighborhood_index',
                                        'LastSaleDate_index'],
                             outputCols=['propertytype_vec', 'siteadress_vec', 'sitecity_vec', 'Homestead_vec', 'Neighborhood_vec', 'LastSaleDate_vec'],
                             handleInvalid='keep')


# COMMAND ----------

# Vector assembler is used to create a vector of input features
 
assembler = VectorAssembler(inputCols=['propertytype_vec', 'siteadress_vec', 'sitecity_vec', 'Homestead_vec', 'Neighborhood_vec', 'LastSaleDate_vec'],
                            outputCol="features")


# COMMAND ----------

# Pipeline is used to pass the data through indexer and assembler simultaneously. Also, it helps to pre-rocess the test data
# in the same way as that of the train data. It also 
 
pipe = Pipeline(stages=[ propertytype_indexer, siteaddress_indexer, sitecity_indexer, homestead, neighborhood_indexer, lastsaledate_indexer,
                       
                        data_encoder,assembler])

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler,StringIndexer
from pyspark.ml import Pipeline


# COMMAND ----------

# Create an object for the Linear Regression model
 
lr_model = LinearRegression(labelCol='propertytype_index')

# COMMAND ----------

# Fit the model on the train data
fit_model = lr_model.fit(train_data.select(['features','propertytype_index']))

# COMMAND ----------

# Transform test data
test_data = fitted_pipe.transform(test_data)
display(test_data)

# COMMAND ----------

# Store the results in a dataframe
results = fit_model.transform(test_data)
display(results)

# COMMAND ----------

results.select(['propertytype_index','prediction']).show()

# COMMAND ----------


test_results = fit_model.evaluate(test_data)

# COMMAND ----------

test_results.residuals.show()

# COMMAND ----------

test_results.rootMeanSquaredError

# COMMAND ----------

test_results.r2
