import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector

df = pd.read_csv("./data_sets/car_mpg.csv")
print(df.head())
data = df.to_numpy()

y = data[:, 1]
x = data[:, 2:8]
print(x. shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)

model_2_features = LinearRegression()
sfs = SequentialFeatureSelector(model_2_features, n_features_to_select = 2)
sfs.fit(x_train, y_train)
print(sfs.get_support())

x_train_selected = sfs.transform(x_train)
x_test_selected = sfs.transform(x_test)
model_2_features.fit(x_train_selected, y_train)
y_pred = model_2_features.predict(x_test_selected)
print(model_2_features.score(x_test_selected, y_test))

model_6_features = LinearRegression()
model_6_features.fit(x_train, y_train)
y_pred = model_6_features.predict(x_test)
print(model_6_features.score(x_test, y_test))