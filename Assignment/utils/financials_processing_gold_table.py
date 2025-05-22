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

def process_labels_gold_table(snapshot_date_str, silver_features_financials_directory, gold_label_store_financials_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_features_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_financials_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # Extract years and months from Credit_History_Age
    def extract_years_months(age_str):
        if age_str is None:
            return None, None
        parts = age_str.split(" ")
        years = int(parts[0]) if parts[0].isdigit() else 0
        months = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
        return years, months    
    extract_udf = udf(extract_years_months, "struct<years:int,months:int>")
    df = df.withColumn("credit_history", extract_udf(col("Credit_History_Age"))) \
           .withColumn("Credit_History_Years", col("credit_history.years").cast(IntegerType())) \
           .withColumn("Credit_History_Months", col("credit_history.months").cast(IntegerType())) \
           .drop("credit_history")
    df = df.drop("Credit_History_Age")
    
    # Calculate EMI to Income Ratio
    df = df.withColumn("EMI_to_Income_Ratio", col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))

    # Split the "Type_of_Loan" column into an array of loan types
    df = df.withColumn("Loan_Types", split(col("Type_of_Loan"), ", and |, "))
    # Get unique loan types
    unique_loan_types = df.select(explode(col("Loan_Types")).alias("Loan_Type")).distinct().collect()
    unique_loan_types = [row["Loan_Type"] for row in unique_loan_types]
    # Create binary columns for each unique loan type
    for loan_type in unique_loan_types:
        df = df.withColumn(loan_type, array_contains(col("Loan_Types"), loan_type).cast("int"))
    # Optionally, drop the temporary "Loan_Types" column
    df = df.drop("Loan_Types", "Type_of_Loan")

    # Process Credit_Mix: Replace None with 'unknown' and create binary columns
    df = df.withColumn("Credit_Mix", when(col("Credit_Mix").isNull(), "unknown").otherwise(col("Credit_Mix")))
    
    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df