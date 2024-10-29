import numpy as np

P = 5
X = np.array([
    [-0.4, -0.4, 0.1, 0.1, 0.6],
    [0.4, -0.1, -0.6, 0.4, -0.1]
])
M = (1/P)*np.matmul(X, X.T)
print(M)

D, V = np.linalg.eig(M)

print(D)
print(V)