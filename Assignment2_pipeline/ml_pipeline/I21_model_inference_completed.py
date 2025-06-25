import os
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import argparse
import pprint
import pickle

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
config["model_date"] =  config['snapshot_date'] - relativedelta(months = 1)
config["model_date_str"] = datetime.strftime(config["model_date"], "%Y-%m-%d")
config["model_name"] = "credit_model_"+config["model_date_str"].replace('-','_')+'.pkl'
config["model_bank_directory"] = "../model_bank/"
config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]

pprint.pprint(config)

gold_directory = f"../datamart/gold/model_predictions/{config['model_name'][:-4]}/"
partition_name = config["model_name"][:-4] + "_predictions_" + current_date_str.replace('-','_') + '.parquet'
filepath = gold_directory + partition_name
if not os.path.exists(filepath):
    print(f"Model predictions directory '{filepath}' does not exist.")
    raise SystemExit("Exiting the program due to missing model predictions.")
print(f"Model predictions directory'{filepath}' exists. Model inference completed.")

