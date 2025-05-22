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
from pyspark.sql.functions import col, when, lower, trim, regexp_replace, split, udf, explode, array_contains, regexp_extract
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
    # Credit history in total months
    df = df.withColumn("Credit_History_Months_Total", 
        col("Credit_History_Years") * 12 + col("Credit_History_Months")
    )
    df = df.drop("Credit_History_Age", "Credit_History_Years", "Credit_History_Months")

    # Calculate EMI to Income Ratio
    df = df.withColumn("EMI_to_Income_Ratio", col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))
    df = df.drop("Total_EMI_per_month")

    # Calculate Invested to Income Ratio
    df = df.withColumn("Invested_to_Income_Ratio", col("Amount_invested_monthly") / col("Monthly_Inhand_Salary"))
    df = df.drop("Amount_invested_monthly", "Monthly_Inhand_Salary", "Annual_Income")

    numerical_cols = [
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", 
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", 
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt", 
    "Credit_Utilization_Ratio", "Monthly_Balance", "Credit_History_Months_Total", 
    "EMI_to_Income_Ratio", "Invested_to_Income_Ratio"
]
    # Handle outliers using Winsorization (cap at 1% and 99% quantiles)
    for col_name in numerical_cols:
        quantiles = df.approxQuantile(col_name, [0.01, 0.99], 0.0)
        lower_bound, upper_bound = quantiles[0], quantiles[1]
        df = df.withColumn(col_name, 
            when(col(col_name) < lower_bound, lower_bound)
            .when(col(col_name) > upper_bound, upper_bound)
            .otherwise(col(col_name))
        )
    
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

    # Process Credit_Mix: Replace None with 'unknown'
    df = df.withColumn("Credit_Mix", when(col("Credit_Mix").isNull(), "unknown").otherwise(col("Credit_Mix")))
    # Credit_Mix: good -> 2, standard -> 1, bad -> 0, unknown -> -1
    df = df.withColumn("Credit_Mix_Encoded",
        when(col("Credit_Mix") == "good", 2)
        .when(col("Credit_Mix") == "standard", 1)
        .when(col("Credit_Mix") == "bad", 0)
        .otherwise(-1)
    )
    df = df.drop("Credit_Mix")
    
    # Payment_of_Min_Amount: yes -> 1, no -> 0, nm -> -1
    df = df.withColumn("Payment_of_Min_Amount_Encoded",
        when(col("Payment_of_Min_Amount") == "yes", 1)
        .when(col("Payment_of_Min_Amount") == "no", 0)
        .otherwise(-1)
    )
    df = df.drop("Payment_of_Min_Amount")

    # Extract and map spending_level: high_spent -> 1, low_spent -> 0
    df = df.withColumn('spending_level',
        when(regexp_extract(col('Payment_Behaviour'), r'(high_spent|low_spent)', 1) == 'high_spent', 1)
        .otherwise(0)
    )
    # Extract and map value_level: small_value -> 0, medium_value -> 1, large_value -> 2
    df = df.withColumn('value_level',
        when(regexp_extract(col('Payment_Behaviour'), r'(small_value|medium_value|large_value)', 1) == 'small_value', 0)
        .when(regexp_extract(col('Payment_Behaviour'), r'(small_value|medium_value|large_value)', 1) == 'medium_value', 1)
        .otherwise(2)
    )
    df = df.drop("Payment_Behaviour")
    
    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df