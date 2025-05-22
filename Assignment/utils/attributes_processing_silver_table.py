import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, trim, regexp_replace, split, udf, explode, array_contains
from pyspark.sql.types import FloatType, IntegerType, DateType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

def process_silver_table(snapshot_date_str, bronze_features_attributes_directory, silver_features_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_features_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Drop rows with null values in critical columns
    critical_columns = ["Customer_ID", "snapshot_date"]
    df = df.na.drop(subset=critical_columns)

    # Convert snapshot_date to DateType
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # Standardize text fields
    text_columns = ["Name", "Occupation"]
    for col_name in text_columns:
        df = df.withColumn(col_name, lower(trim(col(col_name))))

    # Replace "_" with null and handle invalid values
    for column in df.columns:
        df = df.withColumn(column, when(col(column) == "_", None).otherwise(col(column)))
        df = df.withColumn(column, regexp_replace(col(column), "_$", "").cast("string"))

    # Clean Age column: remove trailing characters and cast to Integer
    df = df.withColumn("Age", regexp_replace(col("Age"), "[^0-9]", "").cast(IntegerType()))
    # Handle anomalies in Age (e.g., negative or unrealistic values)
    df = df.withColumn("Age", when((col("Age") < 0) | (col("Age") > 120), None).otherwise(col("Age")))
    df = df.na.drop(subset=["Age"])  # Drop rows with invalid Age

    # Handle SSN: Replace "#F%$D@*&8" with null
    df = df.withColumn("SSN", when(col("SSN") == "#F%$D@*&8", None).otherwise(col("SSN")))

    # Replace "______" in Occupation with null and drop rows with null Occupation
    df = df.withColumn("Occupation", when(col("Occupation") == "______", None).otherwise(col("Occupation")))
    df = df.na.drop(subset=["Occupation"])  # Drop rows with invalid Occupation

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df