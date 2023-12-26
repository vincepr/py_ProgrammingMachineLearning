import numpy as np

def sigmoid(z):
    return 1 /  (1 + np.exp(-z))

# process in moving data trough the system, aka forward propagation.
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

# the old predict function now gets clamped between 0-1 to represent the boolean 
def classify(X, w):
    return np.round(forward(X, w))

# because of local minima we must smooth out ouür loss function. (log loss)
# Y is always 1 or 0 -> so one of the first_term or second_term gets multiplied with 0
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return - np.average(first_term + second_term)

def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    print("completed with w = %s" % w)
    return w

def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess for %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))

# preparing the data:
x1, x2, x3, y = np.loadtxt("./1_perceptron/police.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=10000, lr=0.001)

# testing our model
test(X, Y, w)

# results for 10000  iterations => 25/30 correct
# results for 100000 iterations => 29/30 correct