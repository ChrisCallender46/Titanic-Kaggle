import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer

passengers = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test_ids = test["PassengerId"]

"""
def feature_transformations(data):
    data["cabin_multiple"] = data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(" ")))
    data["cabin_adv"] = data.Cabin.apply(lambda x: str(x)[0])
    data["numeric_ticket"] = data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    data["name_title"] = data.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())

    honorable_names = ["Master", "Dr", "Rev", "Col", "Major", "Sir",
                       "Capt", "the Countess", "Jonkheer"]

    data["name_honor"] = data.name_title.apply(lambda x: 1 if x in honorable_names else 0)

    data["family"] = data["SibSp"] + data["Parch"]

    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId", "name_title"], axis=1)

    return data

feature_transformer = Pipeline([
    ('train_transformer', FunctionTransformer(feature_transformations(passengers))),
    ("test_transformer", FunctionTransformer(feature_transformations(test)))
])

passengers = feature_transformer.fit_transform(passengers)
test = feature_transformer.fit_transform(test)

print(passengers.head().to_string())
print(test.head().to_string())
"""








y = passengers["Survived"]
X = passengers.drop(["Survived"], axis=1)

# Select categorical columns
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
print(categorical_cols)

# Select numerical columns
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
print(numerical_cols)

# Get names of columns with missing values
cols_with_missing = [col for col in X.columns
                     if X[col].isnull().any()]
print(cols_with_missing)

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy="median"))
])

# Preprocessing for categorical data
# Note: Also try a label encoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
])

# Join the pipelines together
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Note: Also try different classification models (Logistic, RFC, GBC, etc)
model = SVC(random_state=42)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
predictions = my_pipeline.predict(X_test)

submission_predictions = my_pipeline.predict(test)

df = pd.DataFrame({"PassengerId": test_ids.values,
                   "Survived": submission_predictions})

df.to_csv("titanic_kaggle_submission.csv", index=False)
