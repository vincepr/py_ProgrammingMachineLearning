# reading and parsing data from minst collection to usable matrix's
import numpy as np
import gzip
import struct


### prepares image data
# read the file of compressed images into a matrix
def load_images(filename):
    with gzip.open(filename, "rb") as f:
        _ignored, n_images, columns, rows = struct.unpack(">IIII", f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # reshape the pixels into a matrix where each line is one image flattened out
        return all_pixels.reshape(n_images, columns * rows)

# insert a column of of 1s for bias
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

# 60000 images, 785 elements ( 1bias + 28*28 px)
# - squashed into one 60000x784 matrix
X_train = prepend_bias(load_images("./ch6-8/train-images-idx3-ubyte.gz"))
# 10000 images, 785 elements ( 1bias + 28*28 px) - so we dont train on the same data we evalute against
# - squashed into one 10000x784 matrix
X_test = prepend_bias(load_images("./ch6-8/t10k-images-idx3-ubyte.gz"))


### prepares minst-labels
def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        f.read(8) # header bytes
        all_labels = f.read()
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)

# we only try to check if were a 5->true or any other number ->false
# - this works only for a binaary check
def encode_fives(Y):
    return(Y == 5).astype(int)

# we extend to a matrix of Y's where we put a a 1 rest 0s for each number once
def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y

## the version where we just check if its a 5 or not:
# 60k labels, each with value 1 if digit is five, 0 otherwise
Y_train = encode_fives(load_labels("./ch6-8/train-labels-idx1-ubyte.gz"))
# 1000 labels, with the same encoding as the training data
Y_test = encode_fives(load_labels("./ch6-8/t10k-labels-idx1-ubyte.gz"))

## the version where we extend for multiple Y values (instead of just checking for 5)
Y_train_unencoded = load_labels("./ch6-8/train-labels-idx1-ubyte.gz")
Y_train_v2 = one_hot_encode(Y_train_unencoded)
Y_test_v2 = load_labels("./ch6-8/t10k-labels-idx1-ubyte.gz")

# Result: Sucess: 9637/10000 (96.37%)