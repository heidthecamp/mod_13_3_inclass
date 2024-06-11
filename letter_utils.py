from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def preprocess_data(letter_df):
    """
    Written for letter data; will split into training
    and testing sets. Uses letter as the target column.
    """
    X = letter_df.drop(columns='lettr')
    y = letter_df['lettr'] # .values.reshape(-1, 1)

    le = LabelEncoder().fit(y)
    y = le.transform(y)

    print(X.head())
    print(y)

    return train_test_split(X, y)

def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def check_metrics(X_test, y_test, model):
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    print(f"Adjusted R-squared: {r2_adj(X_test, y_test, model)}")

    return r2_adj(X_test, y_test, model)

def get_best_pipeline(pipeline1, pipeline2, letter_df):
    """
    Accepts two pipelines and letter data.
    Uses two different preprocessing functions to 
    split the data for training the different 
    pipelines, then evaluates which pipeline performs
    best.
    """
    p1 = pipeline1[0]
    p2 = pipeline2[0]
    X_train, X_test, y_train, y_test = preprocess_data(letter_df)
    p1.fit(X_train, y_train)
    p2.fit(X_train, y_train)
    return pipeline1 if check_metrics(X_test, y_test, p1) > check_metrics(X_test, y_test, p2) else pipeline2

def model_generator(letter_df):
    """
    Generates a pipeline for a linear regression model.
    """
    steps = [
        ('scaler', StandardScaler()),
        ('one_hot', OneHotEncoder(handle_unknown='ignore',
            dtype=int,
            sparse_output=False)),
    ]

    pipelines = [] 
    pipelines.append([Pipeline(steps.copy() + [('model', LogisticRegression())]), 'Logistic Regression'])
    pipelines.append([Pipeline(steps.copy() + [('model', SVC(kernel='linear'))]), 'SVC'])
    pipelines.append([Pipeline(steps.copy() + [('model', KNeighborsClassifier())]), 'KNeighborsClassifier'])
    pipelines.append([Pipeline(steps.copy() + [('model', DecisionTreeClassifier())]), 'DecisionTreeClassifier'])
    pipelines.append([Pipeline(steps.copy() + [('model', RandomForestClassifier())]), 'RandomForestClassifier'])
    pipelines.append([Pipeline(steps.copy() + [('model', XGBClassifier())]), 'XGBClassifier'])

    best = None

    for pipeline in pipelines:

        if best is None:
            best = pipeline
            continue
        else:
            print(f"{best[1]} vs {pipeline[1]}")
            best = get_best_pipeline(best, pipeline, letter_df)


    print(f"Best pipeline: {best[1]}")

    return best[0]

