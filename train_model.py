import os
import pprint
import pandas as pd
import json
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
import onnxruntime as rt
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.data_types import Int64TensorType

titanic_url = (
    "https://raw.githubusercontent.com/amueller/"
    "scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv"
)
data = pd.read_csv(titanic_url)
X = data.drop("survived", axis=1)
y = data["survived"]
print(data.dtypes)

# SimpleImputer on string is not available for
# string in ONNX-ML specifications.
# So we do it beforehand.
for cat in ["embarked", "sex", "pclass"]:
    X[cat].fillna("missing", inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = Pipeline(
    steps=[
        # --- SimpleImputer is not available for strings in ONNX-ML specifications.
        # ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="lbfgs")),
    ]
)


clf.fit(X_train, y_train)

def convert_dataframe_schema(df, drop=None):
    inputs = []
    schema={}
    for k, v in zip(df.columns, df.dtypes):
        if drop is not None and k in drop:
            continue
        if v == 'int64':
            schema[k]='int64'
            t = Int64TensorType([None, 1])
        elif v == 'float64':
            schema[k]='float32'
            t = FloatTensorType([None, 1])
        else:
            schema[k]='string'
            t = StringTensorType([None, 1])
        inputs.append((k, t))
    return inputs, schema

to_drop = {'parch', 'sibsp', 'cabin', 'ticket',
           'name', 'body', 'home.dest', 'boat'}
initial_inputs, schema = convert_dataframe_schema(X_train, to_drop)

pprint.pprint(initial_inputs)

pprint.pprint(X_train[categorical_features+numeric_features].head(3))
print(schema)
with open("schema.json", "w") as sch:
    json.dump(schema, sch)

try:
    model_onnx = convert_sklearn(clf, 'pipeline_titanic', initial_inputs,
                                 target_opset=12, options={id(clf): {'zipmap': False}})
except Exception as e:
    print(e)

with open("pipeline_titanic.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())