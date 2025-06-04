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

def process_labels_gold_table(snapshot_date_str, silver_features_attributes_directory, gold_label_store_attributes_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_features_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_attributes_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # Create Age Group feature
    df = df.withColumn("Age_Group",
        when(col("Age") < 18, "<18")
        .when((col("Age") >= 18) & (col("Age") <= 30), "18-30")
        .when((col("Age") >= 31) & (col("Age") <= 45), "31-45")
        .when((col("Age") >= 46) & (col("Age") <= 60), "46-60")
        .otherwise(">60")
    )

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df