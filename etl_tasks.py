import logging
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.functions import lower, when
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType
from pyspark.sql.functions import mode, mean
from pyspark.ml.feature import Imputer
import psycopg2
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from sklearn.preprocessing import LabelEncoder
import pickle


def preprocess_and_load_to_postgres():
    # Provide the correct path to the necessary JAR file
    jdbc_driver_path = r"/mnt/f/postgresql-42.7.3.jar"

    # Create SparkSession with the correct configuration
    spark = SparkSession.builder \
        .appName("Data Preprocessing") \
        .config("spark.driver.extraClassPath", jdbc_driver_path) \
        .config("spark.executor.extraClassPath", jdbc_driver_path) \
        .getOrCreate()

    # Load training data
    df_train = spark.read.csv('/mnt/f/Train.csv', header=True, inferSchema=True)
    
    # Load test data
    df_test = spark.read.csv('/mnt/f/Test.csv', header=True, inferSchema=True)

    # Replace missing values in Item_Weight column by mean
    imputer = Imputer(inputCols=["Item_Weight"], outputCols=["Item_Weight_imputed"])
    imputer_model = imputer.fit(df_train)
    df_train = imputer_model.transform(df_train).drop("Item_Weight")
    df_train = df_train.withColumnRenamed("Item_Weight_imputed", "Item_Weight")

    imputer_model_test = imputer.fit(df_test)
    df_test = imputer_model_test.transform(df_test).drop("Item_Weight")
    df_test = df_test.withColumnRenamed("Item_Weight_imputed", "Item_Weight")

    # Convert Outlet_Size to numeric values
    df_train = df_train.withColumn("Outlet_Size_numeric",
                                   when(col("Outlet_Size") == "Small", 1)
                                   .when(col("Outlet_Size") == "Medium", 2)
                                   .when(col("Outlet_Size") == "High", 3)
                                   .otherwise(None).cast(IntegerType()))

    df_test = df_test.withColumn("Outlet_Size_numeric",
                                 when(col("Outlet_Size") == "Small", 1)
                                 .when(col("Outlet_Size") == "Medium", 2)
                                 .when(col("Outlet_Size") == "High", 3)
                                 .otherwise(None).cast(IntegerType()))

    # Replace missing values in Outlet_Size column by most occurred value
    imputer = Imputer(inputCols=["Outlet_Size_numeric"], outputCols=["Outlet_Size_imputed"], strategy="mode")
    imputer_model = imputer.fit(df_train)
    df_train = imputer_model.transform(df_train).drop("Outlet_Size", "Outlet_Size_numeric")
    df_train = df_train.withColumnRenamed("Outlet_Size_imputed", "Outlet_Size")

    imputer_model_test = imputer.fit(df_test)
    df_test = imputer_model_test.transform(df_test).drop("Outlet_Size", "Outlet_Size_numeric")
    df_test = df_test.withColumnRenamed("Outlet_Size_imputed", "Outlet_Size")

    # Map imputed numeric values back to original categories
    df_train = df_train.withColumn("Outlet_Size",
                                   when(col("Outlet_Size") == 1, "Small")
                                   .when(col("Outlet_Size") == 2, "Medium")
                                   .when(col("Outlet_Size") == 3, "High"))

    df_test = df_test.withColumn("Outlet_Size",
                                 when(col("Outlet_Size") == 1, "Small")
                                 .when(col("Outlet_Size") == 2, "Medium")
                                 .when(col("Outlet_Size") == 3, "High"))

    # Cleaning of data in item_fat_content

    # Convert all labels to lowercase
    df_train = df_train.withColumn('Item_Fat_Content', lower(df_train['Item_Fat_Content']))
    df_test = df_test.withColumn('Item_Fat_Content', lower(df_test['Item_Fat_Content']))

    # Merge similar categories
    df_train = df_train.withColumn('Item_Fat_Content', when(df_train['Item_Fat_Content'] == 'lf', 'low fat')
                                   .when(df_train['Item_Fat_Content'] == 'reg', 'regular')
                                   .otherwise(df_train['Item_Fat_Content']))
    
    df_test = df_test.withColumn('Item_Fat_Content', when(df_test['Item_Fat_Content'] == 'lf', 'low fat')
                                 .when(df_test['Item_Fat_Content'] == 'reg', 'regular')
                                 .otherwise(df_test['Item_Fat_Content']))

    # Drop unnecessary columns
    columns_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year',]

    # Drop columns in training data
    df_train = df_train.drop(*columns_to_drop)

    # Drop columns in test data
    df_test = df_test.drop(*columns_to_drop)

    # Drop null values in training data
    df_train = df_train.na.drop()

    # Drop null values in test data
    df_test = df_test.na.drop()

    # Write cleaned data to PostgreSQL

    # Define PostgreSQL connection properties
    postgres_url = "jdbc:postgresql://localhost:5432/test_db"
    properties = {
                     "user": "postgres",
                    "password": "postgres",
                    "driver": "org.postgresql.Driver"
                }
    
    

    # Write cleaned data to PostgreSQL
    df_train.write.jdbc(url=postgres_url, table="table_train", mode="overwrite", properties=properties)
    df_test.write.jdbc(url=postgres_url, table="table_test", mode="overwrite", properties=properties)

    # Stop Spark session
    spark.stop()


def train_and_evaluate_models(**kwargs):
    # Define PostgreSQL connection properties
    postgres_host = "localhost"
    postgres_port = "5432"
    postgres_db = "test_db"
    postgres_user = "postgres"
    postgres_password = "postgres"

    # Establish connection to PostgreSQL
    conn = psycopg2.connect(
        host=postgres_host,
        port=postgres_port,
        database=postgres_db,
        user=postgres_user,
        password=postgres_password
    )

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # Define table name
    table_train = "table_train"

    # Define the correct column order
    correct_column_order = [
        "Item_Fat_Content", "Item_Visibility", "Item_Type", 
        "Item_MRP", "Outlet_Location_Type", "Outlet_Type", 
        "Item_Outlet_Sales", "Item_Weight", "Outlet_Size"
    ]

    # Quote the column names to handle case sensitivity
    quoted_column_order = [f'"{col}"' for col in correct_column_order]

    # Fetch data in the correct order
    query = f"SELECT {', '.join(quoted_column_order)} FROM {table_train}"
    cur.execute(query)
    rows = cur.fetchall()

    # Close cursor and connection
    cur.close()
    conn.close()

    # Create DataFrame from fetched data with correct column order
    df_train = pd.DataFrame(rows, columns=correct_column_order)

    # Check the data types of the columns
    logging.info("Data types of columns in df_train: %s", df_train.dtypes)

    # Check the first few rows of df_train
    #logging.info("First few rows of df_train: %s", df_train.head())

    # Perform one-hot encoding for 'Outlet_Type' and 'Item_Type' columns
    encoded_df_train = pd.get_dummies(df_train, columns=['Outlet_Type', 'Item_Type'])

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Perform label encoding for 'Outlet_Location_Type', 'Outlet_Size' columns and 'Item_Fat_Content'
    encoded_df_train['Outlet_Location_Type_LabelEncoded'] = label_encoder.fit_transform(encoded_df_train['Outlet_Location_Type'])
    encoded_df_train['Outlet_Size_LabelEncoded'] = label_encoder.fit_transform(encoded_df_train['Outlet_Size'])
    encoded_df_train['Item_Fat_Content_LabelEncoded'] = label_encoder.fit_transform(encoded_df_train['Item_Fat_Content'])

    # Drop the original columns after label encoding
    encoded_df_train.drop(columns=['Outlet_Location_Type', 'Outlet_Size', 'Item_Fat_Content'], inplace=True)

    logging.info('After dropping columns...')
    logging.info(encoded_df_train.head())

    # Separate features and target variable
    X = encoded_df_train.drop(columns=['Item_Outlet_Sales'])
    y = encoded_df_train['Item_Outlet_Sales']

    # Log shapes of X and y
    logging.info("Shape of X: %s", X.shape)
    logging.info("columns of X: %s", X.columns)
    logging.info("Shape of y: %s", y.shape)

    # Splitting the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    # Log shapes of training and testing sets
    logging.info("Shape of X_train: %s", X_train.shape)
    logging.info("Shape of X_test: %s", X_test.shape)
    logging.info("Shape of y_train: %s", y_train.shape)
    logging.info("Shape of y_test: %s", y_test.shape)

    # Initialize variables to keep track of the best model and its performance
    best_model = None
    best_score = float('inf')  # Initialize with a large value for MSE

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # Initialize dictionary to store evaluation metrics
    evaluation_metrics = {}

    # Train and evaluate models
    for name, model in models.items():
        # Train the model on the training data
        model.fit(X_train, y_train)

        # Predict on the testing data using the current model
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        # Store evaluation metrics in the dictionary
        evaluation_metrics[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R-squared': r2
        }

        # Check if current model outperforms the previous best model
        if mse < best_score:
            best_model = model
            best_score = mse

    # Save the best model as a pickle file
    with open('/mnt/f/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Convert evaluation metrics to DataFrame for easy display
    #evaluation_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index')

    # Push evaluation metrics to XCom
    kwargs['ti'].xcom_push(key='evaluation_metrics', value=evaluation_metrics)
    logging.info('Model Training completed')

def visualize_evaluation_metrics(**kwargs):
    ti = kwargs['ti']
    evaluation_metrics = ti.xcom_pull(task_ids='Model_Training', key='evaluation_metrics')
    
    logging.info("Pulled evaluation metrics from XCom: %s", evaluation_metrics)

    if evaluation_metrics is None:
        raise ValueError("evaluation_metrics is None. Ensure it is correctly pushed to XCom in 'train_and_evaluate_models'.")

    # Convert evaluation metrics to DataFrame for easy display
    evaluation_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index')

    models_names = list(evaluation_metrics.keys())
    mae_values = [metrics['MAE'] for metrics in evaluation_metrics.values()]
    mse_values = [metrics['MSE'] for metrics in evaluation_metrics.values()]
    rmse_values = [metrics['RMSE'] for metrics in evaluation_metrics.values()]
    r2_values = [metrics['R-squared'] for metrics in evaluation_metrics.values()]

    # Plot evaluation metrics for each model
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Evaluation Metrics', fontsize=16)

    # MAE
    axes[0, 0].bar(models_names, mae_values, color='skyblue')
    axes[0, 0].set_title('Mean Absolute Error (MAE)')
    axes[0, 0].set_ylabel('MAE')

    # MSE
    axes[0, 1].bar(models_names, mse_values, color='lightgreen')
    axes[0, 1].set_title('Mean Squared Error (MSE)')
    axes[0, 1].set_ylabel('MSE')

    # RMSE
    axes[1, 0].bar(models_names, rmse_values, color='salmon')
    axes[1, 0].set_title('Root Mean Squared Error (RMSE)')
    axes[1, 0].set_ylabel('RMSE')

    # R-squared
    axes[1, 1].bar(models_names, r2_values, color='gold')
    axes[1, 1].set_title('R-squared (R^2)')
    axes[1, 1].set_ylabel('R-squared')

    # Rotate x-axis labels for better readability
    for ax in axes.flat:
        ax.set_xticklabels(models_names, rotation=45, ha='right')

    # Adjust layout to prevent overlapping labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot to a specified location
    plot_path = '/home/sumit/airflow/dags/evaluation_metrics.png'
    plt.savefig(plot_path)
    plt.close(fig)

    # Log the plot location
    logging.info(f'Plot saved to {plot_path}')
    ti.xcom_push(key='plot_path', value=plot_path)
    
    return evaluation_df


