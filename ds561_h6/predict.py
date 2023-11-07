from google.cloud import storage
from datetime import datetime
from google.cloud.sql.connector import Connector

import sqlalchemy
from flask import Flask, request, Response
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

# enumerating the country column
country_mapping = {country: i for i, country in enumerate(df['country'].unique())}
df['country'] = df['country'].map(country_mapping)

# enumerating the file_name column
name_mapping = {file_name: i for i, file_name in enumerate(df['file_name'].unique())}
df['file_name'] = df['file_name'].map(name_mapping)

# enumerating the time column
name_mapping = {time: i for i, time in enumerate(df['time'].unique())}
df['time'] = df['time'].map(name_mapping)

# enumerating the income column
name_mapping = {income: i for i, income in enumerate(df['income'].unique())}
df['income'] = df['income'].map(name_mapping)

# feature extraction of ip_address
df[['ip_1', 'ip_2', 'ip_3', 'ip_4']] = df['ip_address'].str.split('.', expand=True).astype(int)
df = df.drop('ip_address', axis=1)

# one hot encoding remaining categorical columns
data_columns_categorical = list(df.dtypes[df.dtypes == 'object'].index)
data_cleaned = pd.get_dummies(df, columns=data_columns_categorical)

########### Model 1 ###########
X = data_cleaned[['ip_1', 'ip_2', 'ip_3', 'ip_4']]
y = data_cleaned['country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = RandomForestClassifier(n_estimators=50, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy with Best Model:", accuracy)

########### Model 2 ###########
y = data_cleaned['income']
X = data_cleaned.drop(['income'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = RandomForestClassifier(n_estimators=55, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy with Best Model:", accuracy)