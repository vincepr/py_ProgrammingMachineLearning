import numpy as np

###
### NEURAL NETWORKS LOGIC:
###

### using mini-batch gradient descent instead of GD (gradient descent)
### - what we used so far is batch GD, because it batches ALL data into one
###   then runs calculation on that batch

### Notes:
### - mainly mini-batch-gd gives faster feedback.
### - because it converges faster during early iterations. So if a whole run takes hours/days
###     this provides important early feedback about effectiveness of changed hyperparameters etc.

# gets our current batch-slice from the training data
def prepare_batches(X_train, Y_train, batch_size):
    x_batches = []
    y_batches = []
    n_examples = X_train.shape[0]
    for batch in range(0, n_examples, batch_size):
        batch_end = batch + batch_size
        x_batches.append(X_train[batch:batch_end])
        y_batches.append(Y_train[batch:batch_end])
    return x_batches, y_batches

# and we upgraded our train() to use batches instead of all data

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
    return (y_hat, h)

# takes 2 weights applies them after the other to get predictions
# - argmax to get the highest (biggest statistical chance) result out of the result matrix of chances
def classify(X, w1, w2):
    y_hat, _ = forward(X, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)

# now using cross-entropy-loss instead of log-loss. 'measures' distance between prediction to the labels
#  - this implementation is not stabe so might NaN
def loss(Y, y_hat):
    return - np.sum(Y * np.log(y_hat)) / Y.shape[0]

# human readable 'progress-informations'
def report(epoch, batch, X_train, Y_train, X_test, Y_test, w1, w2):
    y_hat, _ = forward(X_train, w1, w2)
    training_loss = loss(Y_train, y_hat)
    classification = classify(X_test, w1, w2)
    accuracy = np.average(classification == Y_test) * 100.0
    print("%5d=epoch|batch=%d > Loss: %.8f, Accuracy: %.2f%%" % (epoch, batch, training_loss, accuracy))

###
### NEURAL NETWORKS TRAINING:
###
    
# helper 
def sigmoid_gradient(sigmoid):
    return np.multiply(sigmoid, (1 - sigmoid))

## Backpropagation - a way to calculate the gradient of a neural network
## - walk back the graph, for each operation along the way calculate the local gradient
##   multiply all the local gradients together
## note: that is why softmax + cross-entropy work so well together. Their derivative is an easy ŷ-y
def back(X, Y, y_hat, w2, h):
    w2_gradient = np.matmul(prepend_bias(h).T, (y_hat - Y)) / X.shape[0]
    w1_gradient = np.matmul(prepend_bias(X).T, np.matmul(y_hat - Y, w2[1:].T) * sigmoid_gradient(h)) / X.shape[0]
    return (w1_gradient, w2_gradient)

## Node initialization:
## - random (to break up symmetr (matrix multiplication with a uniform matrix -> same columns -> network behaves as one-node))
## - small (to speed up training and avoid dead neurons (big values might saturate the sigmoid functitons))
def initialize_weights(n_input_variables, n_hidden_nodes, n_classes):
    w1_rows = n_input_variables + 1
    w1 = np.random.randn(w1_rows, n_hidden_nodes) * np.sqrt(1 / w1_rows)
    w2_rows = n_hidden_nodes + 1
    w2 = np.random.randn(w2_rows, n_classes) * np.sqrt(1 / w2_rows)
    return (w1, w2)

# for mini-batch-gradient-descent:
# - there are now 2 nested loops.
#   - the outer loop is called epoch and goes trough all batches of the training set
#   - the inner loop is gradient descent on a single batch
# epochs was the old iterations
# and a more batches will help converge w1, w2 earlier
def train(X_train, Y_train, X_test, Y_test, n_hidden_nodes, epochs, batch_size, lr):
    n_input_variables = X_train.shape[1]
    n_classes = Y_train.shape[1]
    w1, w2 = initialize_weights(n_input_variables, n_hidden_nodes, n_classes)
    x_batches, y_batches = prepare_batches(X_train, Y_train, batch_size)
    for epoch in range(epochs):
        for batch in range(len(x_batches)):
            y_hat, h = forward(x_batches[batch], w1, w2)
            w1_gradient, w2_gradient = back(x_batches[batch], y_batches[batch], y_hat, w2, h)
            w1 = w1 - (w1_gradient * lr)
            w2 = w2 - (w2_gradient * lr)
            report(epoch, batch, X_train, Y_train, X_test, Y_test, w1, w2)
    return (w1, w2)

if __name__ == "__main__":
    import mnist as data
    w1, w2 = train(data.X_train, data.Y_train, data.X_test, data.Y_test, 
                   n_hidden_nodes=200, 
                   epochs=2,
                   batch_size=256, # with roughly 60000 sets of training-data a value of 2000 would mean 3 batches
                   lr=0.01)
