<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETL Pipeline Documentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #333;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border: 1px solid #ddd;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>ETL Pipeline Documentation</h1>

    <h2>Introduction</h2>
    <p>
        This document provides clear instructions for setting up and running the ETL (Extract, Transform, Load) pipeline for processing sales data. The pipeline extracts data from CSV files, performs necessary transformations using Apache Spark, loads the transformed data into a PostgreSQL database, trains machine learning models, and evaluates model performance. Additionally, it includes instructions for deploying the trained model using Streamlit for local web-based visualization.
    </p>

    <h2>Prerequisites</h2>
    <p>Before setting up the ETL pipeline, ensure you have the following prerequisites installed:</p>
    <ul>
        <li>Apache Spark</li>
        <li>PostgreSQL</li>
        <li>Python with necessary libraries (<code>pandas</code>, <code>pyspark</code>, <code>scikit-learn</code>, <code>matplotlib</code>, <code>psycopg2</code>)</li>
        <li>Apache Airflow (for scheduling, if desired)</li>
        <li>Streamlit (for model deployment)</li>
    </ul>

    <h2>Setting Up Data Sources</h2>
    <ol>
        <li><strong>Data Files</strong>: Ensure that the CSV data files (<code>Train.csv</code> and <code>Test.csv</code>) are available in the specified directory.</li>
        <li><strong>PostgreSQL Database</strong>: Set up a PostgreSQL database to store the transformed data. Update the database connection details in the pipeline code accordingly.</li>
    </ol>

    <h2>Configuring the ETL Pipeline</h2>
    <ol>
        <li><strong>Pipeline Code</strong>: Review the provided Python code for the ETL pipeline (<code>bigmart_sales_dag.py</code>). This code contains the definition of tasks, dependencies, and execution logic.</li>
        <li><strong>ETL Functions</strong>: Review the <code>etl_tasks.py</code> file containing functions for data preprocessing, model training, evaluation, and deployment. Ensure that the functions are correctly defined and handle data as intended.</li>
    </ol>

    <h2>Running the ETL Pipeline</h2>
    <ol>
        <li><strong>Execute the Pipeline</strong>: Run the Airflow DAG or execute the Python script directly to run the ET
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETL Pipeline Documentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border: 1px solid #ddd;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>ETL Pipeline Documentation</h1>

    <h2>Introduction</h2>
    <p>
        This document provides clear instructions for setting up and running the ETL (Extract, Transform, Load) pipeline for processing sales data. The pipeline extracts data from CSV files, performs necessary transformations using Apache Spark, loads the transformed data into a PostgreSQL database, trains machine learning models, and evaluates model performance. Additionally, it includes instructions for deploying the trained model using Streamlit for local web-based visualization.
    </p>

    <h2>Prerequisites</h2>
    <p>Before setting up the ETL pipeline, ensure you have the following prerequisites installed:</p>
    <ul>
        <li>Apache Spark</li>
        <li>PostgreSQL</li>
        <li>Python with necessary libraries (<code>pandas</code>, <code>pyspark</code>, <code>scikit-learn</code>, <code>matplotlib</code>, <code>psycopg2</code>)</li>
        <li>Apache Airflow (for scheduling, if desired)</li>
        <li>Streamlit (for model deployment)</li>
    </ul>

    <h2>Setting Up Data Sources</h2>
    <ol>
        <li><strong>Data Files</strong>: Ensure that the CSV data files (<code>Train.csv</code> and <code>Test.csv</code>) are available in the specified directory.</li>
        <li><strong>PostgreSQL Database</strong>: Set up a PostgreSQL database to store the transformed data. Update the database connection details in the pipeline code accordingly.</li>
    </ol>

    <h2>Configuring the ETL Pipeline</h2>
    <ol>
        <li><strong>Pipeline Code</strong>: Review the provided Python code for the ETL pipeline (<code>bigmart_sales_dag.py</code>). This code contains the definition of tasks, dependencies, and execution logic.</li>
        <li><strong>ETL Functions</strong>: Review the <code>etl_tasks.py</code> file containing functions for data preprocessing, model training, evaluation, and deployment. Ensure that the functions are correctly defined and handle data as intended.</li>
    </ol>

    <h2>Running the ETL Pipeline</h2>
    <ol>
        <li><strong>Execute the Pipeline</strong>: Run the Airflow DAG or execute the Python script directly to run the ETL pipeline. The pipeline performs the following steps:
            <ul>
                <li>Loads data from CSV files into Spark DataFrames.</li>
                <li>Preprocesses the data by cleaning, transforming, and encoding categorical features.</li>
                <li>Loads the preprocessed data into a PostgreSQL database.</li>
                <li>Trains machine learning models using the preprocessed data.</li>
                <li>Evaluates model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).</li>
                <li>Deploys the trained model using Streamlit for local web-based visualization.</li>
            </ul>
        </li>
    </ol>

    <h2>Scheduling the Pipeline (Optional)</h2>
    <p>
        If you want to automate the execution of the ETL pipeline at regular intervals, you can use Apache Airflow to schedule the tasks. Follow the instructions provided in the Airflow documentation to set up and configure the DAG for scheduling.
    </p>

    <h2>Conclusion</h2>
    <p>
        With the completion of these steps, you should have a fully functional ETL pipeline for processing sales data, including model deployment for local visualization using Streamlit. The pipeline can be run manually or scheduled to run automatically at specified intervals, ensuring that the data remains up-to-date and readily available for analysis.
    </p>
</body>
</html>
