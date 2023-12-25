import numpy as np

def predict(x, w, b):
    return x * w + b

def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

def train(X, Y, iterations, lr):
    w = 0
    b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss = %.6f" % (i, current_loss))

        if loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w + lr, b) < current_loss:
            w += lr
        if loss(X, Y, w, b- lr) < current_loss:
            b -= lr
        elif loss(X, Y, w, b+ lr) < current_loss:
            b += lr
        else:
            return (w, b)
    raise Exception("Failed to get solution in %d iterations " % iterations)


X, Y = np.loadtxt("./srdata/life-expectancy/life-expectancy-without-country-names.txtc/pizza.txt", skiprows=1, unpack=True)
(w, b) = train(X, Y, iterations=10000, lr=0.01)
print("resulting w = %.3f" % w)
print("resulting b = %.3f" % b)
