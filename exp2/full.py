# Import necessary libraries
from keras.callbacks import TensorBoard
import os
import time
# Set up TensorBoard logging
log_dir = os.path.join("logs", "fit", "model_{}".format(int(time.time())))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)




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
print('shape of x_train_vec is' + str(x_train_vec_total.shape))
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
print('shape of y_train_vec is' + str(y_train_vec_total.shape))
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
#
#
# #2
#
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Conv2D(10,(5,5),activation= 'relu', 
input_shape=(28,28,1)))
model.add(layers.MaxPool2D((2,2))) #Output:12×12×10
model.add(layers.Conv2D(20,(5,5),activation= 'relu')) 
#Output:8×8×20
model.add(layers.MaxPool2D((2,2))) #Output:4×4×20
model.add(layers.Flatten())
model.add(layers.Dense(100,activation='relu')) #Output:100
model.add(layers.Dense(10,activation='softmax')) #Output:10
#print the summary of the model
#model.summary()
#
#
# # 3 .1
from keras import optimizers
model.compile(optimizers.rmsprop_v2.RMSprop(learning_rate=0.0001),
                loss = 'categorical_crossentropy',
                metrics =['accuracy'])
#
# #3.2
history =   model.fit(x_train_vec, y_train_vec,
            batch_size=128, epochs =200,
            validation_data = (x_valid_vec,y_valid_vec),
            callbacks=[tensorboard_callback])  # Add the callback here)
#
# 保存模型
# save model
print("Saving model to disk \n")
model.save('HandWriting.h5')
#mp = "E://HandWriting.h5"
#model.save(mp)
#keras.models.save_model(model, 'HandWriting')
#load model
from keras.models import load_model
model2 = load_model('HandWriting.h5')
model2.summary()
#scores = model2.evaluate(x=x, y=Y)
#print('\n%s : %.2f' % (model2.metrics_names[1], scores[1]*100))
# #5.1
#
"""
import matplotlib.pyplot as plt
epochs = range(20) #20 is the number of epochs
train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']
plt.plot(epochs, train_acc, 'bo',label = 'Training Accuracy')
plt.plot(epochs, valid_acc, 'r', label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# Save the plot to a file
plt.savefig('training_validation_accuracy.png')  # Save as PNG file
plt.show()
"""
# #5.2
#
loss_and_acc = model.evaluate(x_test_vec, y_test_vec)
print('loss = ' + str(loss_and_acc[0]))
print('accuracy = ' + str(loss_and_acc[1]))
