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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
#import tensorflow_decision_forests as tfdf

passengers = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test_ids = test["PassengerId"]
"""
def clean(data):
    data["cabin_multiple"] = data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(" ")))
    data["cabin_adv"] = data.Cabin.apply(lambda x: str(x)[0])
    data["numeric_ticket"] = data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

    data = data.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)

    return data

passengers = clean(passengers)
test = clean(test)
"""
"""
def create_features(data):
    data["cabin_multiple"] = data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(" ")))
    data["cabin_adv"] = data.Cabin.apply(lambda x: str(x)[0])
    data["numeric_ticket"] = data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    data["name_title"] = data.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())

    honorable_names = ["Master", "Dr", "Rev", "Col", "Major", "Sir",
                       "Capt", "the Countess", "Jonkheer"]

    data["name_honor"] = data.name_title.apply(lambda x: 1 if x in honorable_names else 0)

    data["family"] = data["SibSp"] + data["Parch"]

    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId", "name_title", "SibSp", "Parch"], axis=1)

    return data

passengers = create_features(passengers)
test = create_features(test)
"""
y = passengers["Survived"]
X = passengers.drop(["Survived"], axis=1)

"""
# Explore and visualize the data
passengers.hist()
plt.show()
print(passengers.info())
print(passengers.describe())

print(passengers["Ticket"].value_counts())
print(passengers["Cabin"].value_counts())
print(passengers["Name"].value_counts())
print(passengers["PassengerId"].value_counts())

corr_matrix = passengers.corr(numeric_only=True)
print(corr_matrix)

num_cols = ["Age", "SibSp", "Parch", "Fare"]
cat_cols = ["Survived", "Pclass", "Sex", "Ticket", "Cabin", "Embarked"]

for x in num_cols:
    passengers[x].hist()
    plt.title(x)
    plt.show()

for x in cat_cols:
    passengers[x].value_counts().plot(kind="bar")
    plt.show()

sns.heatmap(corr_matrix)
plt.show()
"""

"""
print(passengers["cabin_adv"].value_counts())
print(passengers["numeric_ticket"].value_counts())
print(passengers["ticket_letters"].value_counts())
print(pd.pivot_table(passengers, index="Survived", columns="cabin_adv", values="Name", aggfunc="count"))
print(pd.pivot_table(passengers, index="Survived", columns="numeric_ticket", values="Name", aggfunc="count"))
print(pd.pivot_table(passengers, index="Survived", columns="ticket_letters", values="Name", aggfunc="count"))
"""

# Select categorical columns
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
print(categorical_cols)

# Select numerical columns
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
print(numerical_cols)

"""
# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train[my_cols].copy()
X_test = X_test[my_cols].copy()
"""

# Get names of columns with missing values
cols_with_missing = [col for col in X.columns
                     if X[col].isnull().any()]
print(cols_with_missing)

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy="median"),
     )
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
model = GradientBoostingClassifier()

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

"""
def clean(data):
    data["cabin_multiple"] = data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(" ")))
    data["cabin_adv"] = data.Cabin.apply(lambda x: str(x)[0])
    data["numeric_ticket"] = data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    data["name_title"] = data.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())

    honorable_names = ["Master", "Dr", "Rev", "Col", "Major", "Sir",
                       "Capt", "the Countess", "Jonkheer"]

    data["name_honor"] = data.name_title.apply(lambda x: 1 if x in honorable_names else 0)

    data["family"] = data["SibSp"] + data["Parch"]

    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId", "name_title"], axis=1)

    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)

    data["norm_sibsp"] = np.log(data.SibSp + 1)

    data["norm_fare"] = np.log(data.Fare + 1)

    data["norm_family"] = np.log(data.family + 1)

    data = data.drop(["family", "Fare", "SibSp"], axis=1)

    data.Embarked.fillna("U", inplace=True)
    return data


passengers = clean(passengers)
test = clean(test)

le = preprocessing.LabelEncoder()

cols = ["Sex", "Embarked", "cabin_adv"]

for col in cols:
    passengers[col] = le.fit_transform(passengers[col])
    test[col] = le.transform(test[col])
    print(le.classes_)

y = passengers["Survived"]
X = passengers.drop(["Survived"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=passengers["Survived"], random_state=42)

model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)


#model = RandomForestClassifier(random_state=42)
#model = model.fit(X_train, y_train)


#model = GradientBoostingClassifier(random_state=42)
#model = model.fit(X_train, y_train)


#model = RidgeClassifier(random_state=42)
#model = model.fit(X_train, y_train)

#model = SVC()
#model = model.fit(X_train, y_train)

#model = TransformedTargetRegressor()
#model = model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
print(recall_score(y_test, predictions))

submission_preds = model.predict(test)

df = pd.DataFrame({"PassengerId": test_ids.values,
                   "Survived": submission_preds})

df.to_csv("submission.csv", index=False)
"""
