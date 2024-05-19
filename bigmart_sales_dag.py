from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

# Import functions from etl_tasks.py
from etl_tasks import preprocess_and_load_to_postgres, train_and_evaluate_models, visualize_evaluation_metrics

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Create a DAG object
dag = DAG(
    dag_id='sale_dag',
    default_args=default_args,
    schedule_interval='@daily',  # Set the schedule interval to run daily
    catchup=False  # Optional: prevents the DAG from running for past dates if it wasn't active then
)
# Define tasks for data preprocessing and loading to PostgreSQL
preprocess_task = PythonOperator(
    task_id='Data_Ingestion_and_Preparation',
    python_callable=preprocess_and_load_to_postgres,
    dag=dag,
)

# Define tasks for model training and evaluation
train_and_evaluate_task = PythonOperator(
    task_id='Model_Training',
    python_callable=train_and_evaluate_models,
    dag=dag,
)

# Define task for model visualization
visualize_task = PythonOperator(
    task_id='Model_Evaluation',
    python_callable=visualize_evaluation_metrics,
    dag=dag,
)

# Define task for deploying the model using Streamlit
deploy_model_task = BashOperator(
    task_id='Model_Deployment',
    bash_command=(
        'cd /home/sumit/airflow/dags && export EMAIL_ADDRESS=""; nohup streamlit run --server.port=8502 /home/sumit/airflow/dags/app.py &> '
        '/home/sumit/airflow/dags/streamlit_log.txt &'
    ),
    dag=dag,
)

# Define task dependencies
preprocess_task >> train_and_evaluate_task >> visualize_task >> deploy_model_task
