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
model.summary()

# # 3 .1
from keras import optimizers
model.compile(optimizers.rmsprop_v2.RMSprop(learning_rate=0.00
01),
loss = 'categorical_crossentropy',
metrics =['accuracy'])
#
# #3.2
history = model.fit(x_train_vec, y_train_vec,
batch_size=128, epochs =20,
validation_data = (x_valid_vec,y_valid_vec))
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
#print('\n%s : %.2f' % (model2.metrics_names[1], 
scores[1]*100))