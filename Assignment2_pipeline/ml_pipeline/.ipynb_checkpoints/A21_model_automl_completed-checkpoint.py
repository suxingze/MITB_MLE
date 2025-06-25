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
import argparse

# --------------------------------
# Parse Command Line Arguments
# --------------------------------
parser = argparse.ArgumentParser(description="Data pipeline start script for data source checks.")
parser.add_argument('--snapshotdate', type=str, help='The snapshot date for the data checks (YYYY-MM-DD).')
args = parser.parse_args()
current_date_str = args.snapshotdate
current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()


# --- set up config ---
config = {}
config["snapshot_date_str"] = current_date_str
config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
config["model_name"] = "credit_model_"+config["snapshot_date_str"].replace('-','_')+'.pkl'
config["model_bank_directory"] = "../model_bank/"
config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]

pprint.pprint(config)

# --- load model artefact from model bank ---
# Load the model from the pickle file

with open(config["model_artefact_filepath"], 'rb') as file:
    model_artefact = pickle.load(file)

print("Auto model training completed. ")





