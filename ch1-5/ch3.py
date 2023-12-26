import numpy as np

def predict(x, w, b):
    return x * w + b

def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)

def train(X, Y, iterations, lr):
    w = 0
    b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss = %.10f" % (i, current_loss))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b

X, Y = np.loadtxt("./ch1-5/pizza.txt", skiprows=1, unpack=True)
(w, b) = train(X, Y, iterations=20000, lr=0.001)
print("resulting w = %.10f" % w)
print("resulting b = %.10f" % b)
print("Predicting x=%d => y = %.3f" % (20, predict(20, w, b)))
