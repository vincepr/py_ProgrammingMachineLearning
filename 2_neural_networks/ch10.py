import numpy as np

### NEURAL NETWORKS LOGIC:

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# we want all our results to add up to 1. So we can interpret them as statistical weights
def softmax(logits):
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)

# adds a column of 1s for our bias
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

# forward propagation - calculates the system's outputs from the systems inputs
# - 1. add a bias column
# - 2. compute the weighted sum of the inputs using the first matrix of weights w1
# - 3. pass the result trough the sigmoid function
#      = we calculated the hidden layer
# - 4-5 we use a different matrix of weights w2 and and pass the weighted sum trough
# - 6. the result trough softmax instead of sigmoid.
def forward(X, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(X), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return y_hat

# takes 2 weights applies them after the other to get predictions
# - argmax to get the highest (biggest statistical chance) result out of the result matrix of chances
def classify(X, w1, w2):
    y_hat = forward(X, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)

# now using cross-entropy-loss instead of log-loss. 'measures' distance between prediction to the labels
def loss(Y, y_hat):
    return - np.sum(Y * np.log(y_hat)) / Y.shape[0]

# human readable 'progress-informations'
def report(iteration, X_train, Y_train, X_test, Y_test, w1, w2):
    y_hat = forward(X_train, w1, w2)
    training_loss = loss(Y_train, y_hat)
    classification = classify(X_test, w1, w2)
    accuracy = np.average(classification == Y_test) * 100.0
    print("Iteration: %5d, Loss: %.6f, Accuracy: %.2ff%%" % (iteration, training_loss, accuracy))

### using some precompiled test-data
import json
with open("./2_neural_networks/ch10_weights.json") as f:
    weights = json.load(f)
w1, w2 = (np.array(weights[0]), np.array(weights[1]))
import mnist
report(0, mnist.X_train, mnist.Y_train, mnist.X_test, mnist.Y_test, w1, w2)
# 43.19% with just a few training rounds

### NEURAL NETWORKS TRAINING: