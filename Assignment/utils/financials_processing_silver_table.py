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

def process_silver_table(snapshot_date_str, bronze_features_financials_directory, silver_features_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_features_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Handle anomalies (e.g., negative values where not allowed)
    df = df.withColumn("Annual_Income", when(col("Annual_Income") < 0, None).otherwise(col("Annual_Income"))) \
           .withColumn("Monthly_Inhand_Salary", when(col("Monthly_Inhand_Salary") < 0, None).otherwise(col("Monthly_Inhand_Salary"))) \
           .withColumn("Num_Bank_Accounts", when(col("Num_Bank_Accounts") < 0, None).otherwise(col("Num_Bank_Accounts"))) \
           .withColumn("Num_Credit_Card", when(col("Num_Credit_Card") < 0, None).otherwise(col("Num_Credit_Card"))) \
           .withColumn("Interest_Rate", when(col("Interest_Rate") < 0, None).otherwise(col("Interest_Rate"))) \
           .withColumn("Num_of_Loan", when(col("Num_of_Loan") < 0, None).otherwise(col("Num_of_Loan")))
    df = df.withColumn("Num_Bank_Accounts", when(col("Num_Bank_Accounts") > 100, None).otherwise(col("Num_Bank_Accounts"))) \
           .withColumn("Num_Credit_Card", when(col("Num_Credit_Card") > 100, None).otherwise(col("Num_Credit_Card"))) \
           .withColumn("Interest_Rate", when(col("Interest_Rate") > 35, None).otherwise(col("Interest_Rate")))
    df = df.na.drop()

    # Replace "_" with null and handle invalid numeric values
    for column in df.columns:
        df = df.withColumn(column, when(col(column) == "_", None).otherwise(col(column)))
        df = df.withColumn(column, regexp_replace(col(column), "_$", "").cast("string"))

    # Correct data types
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType())) \
           .withColumn("Annual_Income", col("Annual_Income").cast(FloatType())) \
           .withColumn("Monthly_Inhand_Salary", col("Monthly_Inhand_Salary").cast(FloatType())) \
           .withColumn("Num_Bank_Accounts", col("Num_Bank_Accounts").cast(IntegerType())) \
           .withColumn("Num_Credit_Card", col("Num_Credit_Card").cast(IntegerType())) \
           .withColumn("Interest_Rate", col("Interest_Rate").cast(FloatType())) \
           .withColumn("Num_of_Loan", col("Num_of_Loan").cast(IntegerType())) \
           .withColumn("Delay_from_due_date", col("Delay_from_due_date").cast(IntegerType())) \
           .withColumn("Num_of_Delayed_Payment", col("Num_of_Delayed_Payment").cast(IntegerType())) \
           .withColumn("Changed_Credit_Limit", col("Changed_Credit_Limit").cast(FloatType())) \
           .withColumn("Num_Credit_Inquiries", col("Num_Credit_Inquiries").cast(FloatType())) \
           .withColumn("Outstanding_Debt", col("Outstanding_Debt").cast(FloatType())) \
           .withColumn("Credit_Utilization_Ratio", col("Credit_Utilization_Ratio").cast(FloatType())) \
           .withColumn("Total_EMI_per_month", col("Total_EMI_per_month").cast(FloatType())) \
           .withColumn("Amount_invested_monthly", col("Amount_invested_monthly").cast(FloatType())) \
           .withColumn("Monthly_Balance", col("Monthly_Balance").cast(FloatType()))

    # Drop rows with null values in critical columns
    critical_columns = ["Customer_ID", "snapshot_date", "Annual_Income", "Monthly_Inhand_Salary", "Total_EMI_per_month", "Amount_invested_monthly", "Changed_Credit_Limit"]
    df = df.na.drop(subset=critical_columns)
    
    # Standardize text fields
    text_columns = ["Type_of_Loan", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
    for col_name in text_columns:
        df = df.withColumn(col_name, lower(trim(col(col_name))))

    # Remove rows with abnormal value '!@9#%8' in Payment_Behaviour
    df = df.filter(col("Payment_Behaviour") != '!@9#%8')

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df