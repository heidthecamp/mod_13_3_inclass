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
    print(y.head())

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
    X_train, X_test, y_train, y_test = preprocess_data(letter_df)
    pipeline1.fit(X_train, y_train)
    pipeline2.fit(X_train, y_train)
    print("Pipeline 1:")
    check_metrics(X_test, y_test, pipeline1)
    print("Pipeline 2:")
    check_metrics(X_test, y_test, pipeline2)
    return pipeline1 if check_metrics(X_test, y_test, pipeline1) > check_metrics(X_test, y_test, pipeline2) else pipeline2

def model_generator(letter_df):
    """
    Generates a pipeline for a linear regression model.
    """
    # return Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('one_hot', OneHotEncoder(handle_unknown='ignore',
    #                             sparse_output=False)),
    # ])

    steps = [
        ('scaler', StandardScaler()),
        ('one_hot', OneHotEncoder(handle_unknown='ignore',
            dtype=int,
            sparse_output=False)),
    ]

    # pipelines = [] 

    # pipelines.append([Pipeline(steps.copy()), 'Logistic Regression'])
    # pipelines[-1].steps.append(('model', LogisticRegression()))
    # pipelines.append([Pipeline(steps.copy()), ])
    # pipelines[-1].steps.append(('model', SVC(kernel='linear')))
    # # pipelines.append(Pipeline(steps.copy()))
    # # pipelines[-1].steps.append(('model', KNeighborsClassifier()))
    # pipelines.append(Pipeline(steps.copy()))
    # pipelines[-1].steps.append(('model', DecisionTreeClassifier()))
    # pipelines.append(Pipeline(steps.copy()))
    # pipelines[-1].steps.append(('model', RandomForestClassifier()))
    # pipelines.append(Pipeline(steps.copy()))
    # pipelines[-1].steps.append(('model', XGBClassifier()))

    pipeline1 = Pipeline(steps.copy())
    pipeline2 = Pipeline(steps.copy())
    # pipeline3 = Pipeline(steps.copy())
    pipeline4 = Pipeline(steps.copy())
    pipeline5 = Pipeline(steps.copy())
    pipeline6 = Pipeline(steps.copy())

    pipeline1.steps.append(('model', LogisticRegression()))
    pipeline2.steps.append(('model', SVC(kernel='linear')))
    # pipeline3.steps.append(('model', KNeighborsClassifier()))
    pipeline4.steps.append(('model', DecisionTreeClassifier()))
    pipeline5.steps.append(('model', RandomForestClassifier()))
    pipeline6.steps.append(('model', XGBClassifier()))          

    best = None

    best = get_best_pipeline(pipeline1, pipeline2, letter_df)
    # best = get_best_pipeline(best, pipeline3, letter_df)
    best = get_best_pipeline(best, pipeline4, letter_df)
    best = get_best_pipeline(best, pipeline5, letter_df)
    best = get_best_pipeline(best, pipeline6, letter_df)

    # for pipeline in pipelines:

    #     if best is None:
    #         best = pipeline
    #         continue
    #     else:
    #         best = get_best_pipeline(best, pipeline, letter_df)

    return best



