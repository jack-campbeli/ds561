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
# df = load_data()
def function1():
    df = pd.read_csv("data.csv")
    print(df.head())
    print(df.columns)

    df['ip_address'] = df['ip_address'].str.replace('.', '').astype('int64')
    print(df['ip_address'])

    # Convert 'country' to numeric labels
    country_mapping = {country: i for i, country in enumerate(df['country'].unique())}
    df['country'] = df['country'].map(country_mapping)

    print(df['country'].nlargest(1))

    # Split the data into features and target
    X = df['ip_address'].values.reshape(-1,1)
    y = df['country']

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    ###### Grid Search ######
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'bootstrap': [True, False],
    #     'criterion': ['gini', 'entropy']
    # }

    # random_forest = RandomForestClassifier(random_state=0)
    # grid_search = GridSearchCV(random_forest, param_grid, cv=5, scoring='accuracy', n_jobs=1)

    # grid_search.fit(X_train, y_train)

    # print()
    # print("Best Hyperparameters:", grid_search.best_params_)
    # print("Best F1 Score:", grid_search.best_score_)

    # best_rf = grid_search.best_estimator_

    # y_pred = best_rf.predict(X_test)

    # accuracy = accuracy_score(y_test, y_pred)

    # print()
    # print("Accuracy with Best Model:", accuracy)

    ###### Random Forest ######
    random_forest = RandomForestClassifier(n_estimators=100,  # You can specify the number of trees
                                        max_depth=None,     # Specify the maximum depth of trees
                                        min_samples_split=2,  # Specify the minimum number of samples required to split an internal node
                                        min_samples_leaf=1,   # Specify the minimum number of samples required to be at a leaf node
                                        max_features='auto',  # Number of features to consider for the best split (you can adjust this)
                                        bootstrap=True,       # Whether to use bootstrapping when building trees
                                        criterion='gini',     # Splitting criterion, can be 'gini' or 'entropy'
                                        random_state=0)       # Seed for random number generator

    # Fit the Random Forest classifier to the training data
    random_forest.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = random_forest.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

def function2():
    df = pd.read_csv("data1.csv")

    # Convert categorical variables to one-hot encoding
    df = pd.get_dummies(df, columns=['gender', 'age', 'time', 'country'], drop_first=True)

    # Separate the target and features for set 2 (x2, y2)
    x2 = df.drop(['income', 'name', 'ip_address'], axis=1)  # Remove 'ip_address'
    y2 = df['income']

    # Split the data into training and testing sets for set 2
    X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, random_state=0)

    # Define a parameter grid for the grid search
    param_grid = {
        'n_estimators': [100, 200, 300],  # You can experiment with different values
        'max_depth': [None, 10, 20, 30],  # You can experiment with different values
        'min_samples_split': [2, 5, 10],  # You can experiment with different values
        'min_samples_leaf': [1, 2, 4],  # You can experiment with different values
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    # Initialize RandomForestClassifier
    random_forest = RandomForestClassifier(random_state=0)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(random_forest, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train2, y_train2)

    # Print the best hyperparameters and best score
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Accuracy Score:", grid_search.best_score_)

    # Get the best estimator (RandomForestClassifier with the best hyperparameters)
    best_rf = grid_search.best_estimator_

    # Make predictions on the test data using the best model
    y_pred2 = best_rf.predict(X_test2)

    # Calculate accuracy for set 2
    accuracy = accuracy_score(y_test2, y_pred2)
    print("Accuracy for set 2:", accuracy)

function2()