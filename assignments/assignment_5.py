import numpy as np

X = np.array([
    [-0.4]
])

M = np.array([
    [0.7, -0.2],
    [-0.2, 0.7]
])
D, V = np.linalg.eig(M)

print(D, V)