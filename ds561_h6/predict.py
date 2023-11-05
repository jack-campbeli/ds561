from google.cloud import storage
from datetime import datetime
from google.cloud.sql.connector import Connector

import sqlalchemy
from flask import Flask, request, Response
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd

INSTANCE_CONNECTION_NAME = "jacks-project-398813:us-east1:database-1"
DB_USER = "root"
DB_PASS = ""
DB_NAME = "db1"

########### 1. Load data from SQL Server ###########


def load_data():
    try:
        # initialize Connector object
        connector = Connector()

        # function to return the database connection object
        def getconn():
            conn = connector.connect(
                INSTANCE_CONNECTION_NAME,
                "pymysql",
                user=DB_USER,
                password=DB_PASS,
                db=DB_NAME
            )
            return conn

        # create connection pool with 'creator' argument to our connection object function
        pool = sqlalchemy.create_engine(
            "mysql+pymysql://",
            creator=getconn,
        )

        sql_query = sqlalchemy.text("SELECT * FROM requests")

        with pool.connect() as db_conn:
            results = db_conn.execute(sql_query)
            data = results.fetchall()

        columns = results.keys()
        df = pd.DataFrame(data, columns=columns)
        
        if df is not None and not df.empty:
            print("Data retrieved.")
            return df
        else:
            print("No data retrieved from the database.")
            return None

    except Exception as e:
        print(f"Error retrieving data from the database: {str(e)}")

# main
df = load_data()
print(df.head())

data_columns_categorical = list(df.dtypes[df.dtypes == 'object'].index)
clean_df = pd.get_dummies(df, columns=data_columns_categorical)

y = clean_df['country']
X = clean_df['ip_address']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LogisticRegression(max_iter=1000)  # You can adjust hyperparameters as needed
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")