import numpy as np

def predict(X, w):
    return np.matmul(X, w)

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss = %.20f" % (i, current_loss))
        w -= gradient(X, Y, w) * lr
    return w

x1, x2, x3, y = np.loadtxt("./1_perceptron/pizza_3_vars.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=10000, lr=0.001)

print("resulting w = %s" % w.T)
print("\n a few predictions:")
for i in range(5):
    print("X[%d] = %.6f (label: %d)" % (i, predict(X[i], w), Y[i]))
