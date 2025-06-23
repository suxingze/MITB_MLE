from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 7, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---

    dep_check_source_label_data = BashOperator(
        task_id="dep_check_source_label_data",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D00_label_source_check.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D10_bronze_label.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label_store = BashOperator(
        task_id="silver_label_store",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D20_silver_label.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store = BashOperator(
        task_id="gold_label_store",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D30_gold_label.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    label_store_completed = BashOperator(
        task_id="label_store_completed",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D40_label_store_completed.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # Define task dependencies to run scripts sequentially
    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed
 
 
    # --- feature store ---
    dep_check_source_data_bronze_1 = BashOperator(
        task_id="dep_check_source_data_bronze_1",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D01_source_check_bronze_1.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    dep_check_source_data_bronze_2 = BashOperator(
        task_id="dep_check_source_data_bronze_2",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D02_source_check_bronze_2.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    dep_check_source_data_bronze_3 = BashOperator(
        task_id="dep_check_source_data_bronze_3",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D03_source_check_bronze_3.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    bronze_table_1 = BashOperator(
        task_id="bronze_table_1",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D11_bronze_table_1.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    bronze_table_2 = BashOperator(
        task_id="bronze_table_2",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D12_bronze_table_2.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    bronze_table_3 = BashOperator(
        task_id="bronze_table_3",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D13_bronze_table_3.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_1 = BashOperator(
        task_id="silver_table_1",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/21_silver_table_1.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_2 = BashOperator(
        task_id="silver_table_2",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/22_silver_table_2.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_3 = BashOperator(
        task_id="silver_table_3",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/23_silver_table_3.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_feature_store_1 = BashOperator(
        task_id="gold_feature_store_1",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D31_gold_feature_store_1.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_feature_store_2 = BashOperator(
        task_id="gold_feature_store_2",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D32_gold_feature_store_2.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    feature_store_completed = BashOperator(
        task_id="feature_store_completed",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_pipeline/D41_feature_store_completed.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    # Define task dependencies to run scripts sequentially
    dep_check_source_data_bronze_1 >> bronze_table_1 >> silver_table_1 >> gold_feature_store_1
    dep_check_source_data_bronze_2 >> bronze_table_2 >> silver_table_1 >> gold_feature_store_2
    dep_check_source_data_bronze_3 >> bronze_table_3 >> silver_table_2
    gold_feature_store_1 >> feature_store_completed
    gold_feature_store_2 >> feature_store_completed


    # --- model inference ---
    model_inference_start = BashOperator(
        task_id="model_inference_start",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/I01_model_inference_start.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/I11_model_inference.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_inference_completed = BashOperator(
        task_id="model_inference_completed",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/I21_model_inference_completed.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_inference_start
    model_inference_start >> model_inference >> model_inference_completed


    # --- model monitoring ---
    model_monitor_start = BashOperator(
        task_id="model_monitor_start",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/M01_model_monitor_start.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_monitor = BashOperator(
        task_id="model_monitor",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/M11_model_monitor.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_monitor_completed = BashOperator(
        task_id="model_monitor_completed",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/M21_model_monitor_completed.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_monitor >> model_monitor_completed


    # --- model auto training ---

    model_automl_start = BashOperator(
        task_id="model_automl_start",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/A01_model_automl_start.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    model_automl = BashOperator(
        task_id="model_automl",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/A11_model_automl.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    model_automl_completed = BashOperator(
        task_id="model_automl_completed",
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 ml_pipeline/A21_model_automl_completed.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_automl_start
    label_store_completed >> model_automl_start
    model_automl_start >> model_automl >> model_automl_completed
    