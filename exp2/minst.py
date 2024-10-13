#1.1
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
print('shape of x_train:' + str(x_train.shape))
print('shape of x_test:' + str(x_test.shape))
print('shape of y_train:' + str(y_train.shape))
print('shape of y_test:' + str(y_test.shape))
#1.2
x_train_vec_total = x_train.reshape((60000,28,28,1))/255.0
x_test_vec = x_test.reshape((10000,28,28,1))/255.0
print('shape of x_train_vec is' + 
str(x_train_vec_total.shape))
#
# #1.3
import numpy as np
def to_one_hot(labels,dimension = 10):
    results = np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i, label] =1.
    return results
y_train_vec_total = to_one_hot(y_train)
y_test_vec = to_one_hot(y_test)
print('shape of y_train_vec is' + 
str(y_train_vec_total.shape))
#
#
#
# #1.4
#
#
rand_indices = np.random.permutation(60000)
train_indices = rand_indices[0:50000]
valid_indices = rand_indices[50000:60000]
x_train_vec = x_train_vec_total[train_indices, :]
y_train_vec = y_train_vec_total[train_indices, :]
print('shape of x_train_vec: ' + str(x_train_vec.shape))
print('shape of y_train_vec: ' + str(y_train_vec.shape))
x_valid_vec = x_train_vec_total[valid_indices, :]
y_valid_vec = y_train_vec_total[valid_indices, :]
print('shape of x_valid_vec: ' + str(x_valid_vec.shape))
print('shape of y_valid_vec: ' + str(y_valid_vec.shape))