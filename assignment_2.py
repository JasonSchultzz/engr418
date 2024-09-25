from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

t2 = np.array([0.004624, 0.014641, 0.027889, 0.042025, 0.056169, 0.071289, 0.083521, 0.101124, 0.1225, 0.135424])
h = np.array([0.1, 0.2, 0.33, 0.43, 0.54, 0.66, 0.72, 0.85, 0.99, 1.07])

x = h.reshape(-1, 1)
y = t2.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
m = model.coef_[0]
b = model.intercept_
print(m, b)
g = 2/m
print(g)

x_plot = [[0], [1.2]]
y_plot = model.predict(x_plot)
plt.scatter(h, t2)
plt.plot(x_plot, y_plot, "r")
plt.grid()
plt.xlabel("Height (m)")
plt.ylabel("Time Squared ($t^{2}$)")
plt.show()