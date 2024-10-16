'''
ENGR 418 Assignment 3, Question 8
Jason Schultz
24662652

Results that were obtained:
Confusion Matrix:
[[10  0]
 [ 0 10]]
Accuracy Score:
1.0
Model Weights:
w0 = 1.0514
w1 = 0.0874
w2 = -0.1838
w3 = -0.9091
w4 = -0.2711
w5 = 0.0388
w6 = -0.3787
w7 = 1.1775
w8 = -0.3261
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("./assignments/spectrometer_dataset.csv")
print(df.head())
x1 = df["feature 1"]
x2 = df["feature 2"]
x3 = df["feature 3"]
x4 = df["feature 4"]
x5 = df["feature 5"]
x6 = df["feature 6"]
x7 = df["feature 7"]
x8 = df["feature 8"]
x = np.array([x1, x2, x3, x4, x5, x6, x7, x8]).T
y = np.array(df["label"]).reshape(-1, 1)
print(x.shape, y.shape)

model = LogisticRegression()
model.fit(x, y)
y_pred = model.predict(x).reshape(-1, 1)
print(y_pred.shape)

print(model.intercept_, model.coef_)
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print("Accuracy Score:")
print(accuracy_score(y, y_pred))

print("Model Weights:")
print(f"w0 = {model.intercept_[0]:.4f}")
for (i, weight) in enumerate(model.coef_[0]):
    print(f"w{i+1} = {weight:.4f}")