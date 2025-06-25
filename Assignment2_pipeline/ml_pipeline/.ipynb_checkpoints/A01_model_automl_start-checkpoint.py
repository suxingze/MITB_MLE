import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
model_train_date_str = "2024-09-01"
train_test_period_months = 12
oot_period_months = 2
train_test_ratio = 0.8

config = {}
config["model_train_date_str"] = model_train_date_str
config["train_test_period_months"] = train_test_period_months
config["oot_period_months"] =  oot_period_months
config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
config["train_test_ratio"] = train_test_ratio 

# get label store
# connect to label store
folder_path = "../datamart/gold/label_store/"
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
label_store_sdf = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",label_store_sdf.count())

# extract label store
labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))
print("extracted labels_sdf", labels_sdf.count(), config["train_test_start_date"], config["oot_end_date"])

# get feature store
gold_clks_directory = "../datamart/gold/feature_store/eng/"
folder_path = gold_clks_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
feature_store_sdf_1 = spark.read.parquet(*files_list)
print("row_count:",feature_store_sdf_1.count())

gold_fin_directory = "../datamart/gold/feature_store/cust_fin_risk/"
folder_path = gold_fin_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
feature_store_sdf_2 = spark.read.parquet(*files_list)
print("row_count:",feature_store_sdf_2.count())

feature_store_sdf_1 = feature_store_sdf_1.withColumn("snapshot_date", to_date(col("snapshot_date")))
feature_store_sdf_2 = feature_store_sdf_2.withColumn("snapshot_date", to_date(col("snapshot_date")))
feature_store_sdf_1.printSchema()
feature_store_sdf_2.printSchema()

# extract feature store
features_sdf_1 = feature_store_sdf_1.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))
features_sdf_2 = feature_store_sdf_2
features_sdf_2 = features_sdf_2.drop('snapshot_date')
print("extracted features_sdf_1", features_sdf_1.count(), config["train_test_start_date"], config["oot_end_date"])
print("extracted features_sdf_2", features_sdf_2.count(), config["train_test_start_date"], config["oot_end_date"])

# join two feature tables
features_sdf = features_sdf_1.join(features_sdf_2, on=["Customer_ID"], how="left")
print("extracted features_sdf", features_sdf.count(), config["train_test_start_date"], config["oot_end_date"])
features_sdf.show(5)

features_pdf = features_sdf.toPandas()
columns_to_exclude = ['Customer_ID', 'snapshot_date']
columns_to_rename = [col for col in features_pdf.columns if col not in columns_to_exclude]
rename_dict = {col: 'feature_' + col for col in columns_to_rename}
features_pdf.rename(columns=rename_dict, inplace=True)
features_sdf = spark.createDataFrame(features_pdf)
features_sdf.show(5)


