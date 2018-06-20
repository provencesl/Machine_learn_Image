# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:34:45 2018

@author: Beobachter
"""
from keras.models import Sequential
#from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten,Dense
import numpy as np
from keras import regularizers
import matplotlib.pyplot as plt
import keras

def train_fc():
    #导入bottleneck feature
    train_data = np.load(open('bottleneck_feature_train.npy', 'rb'))
    train_labels = np.array([0] * 500 + [1] * 500 + [2] * 500 + [3] * 500)  #打标签
    validation_data = np.load(open('bottleneck_feature_validation.npy', 'rb'))
    validation_labels = np.array([0] * 200 + [1] * 200 + [2] * 200 + [3] * 200)
    
    train_labels = keras.utils.to_categorical(train_labels, 4) #将类别向量映射为二值类别矩阵
    validation_labels = keras.utils.to_categorical(validation_labels, 4)
    
    #构建网络
    model = Sequential()
    model.add(Flatten(input_shape = (7,7,512)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax', kernel_regularizer = regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.001)))
    model.compile(optimizer='rmsprop',   #编译模型
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])
    #训练模型
    hist = model.fit(train_data, train_labels, nb_epoch = 50, batch_size=32,validation_data = (validation_data, validation_labels))
    #评估模型
    score = model.evaluate(validation_data, validation_labels)
    #print('test score:', score[0])
    print('test acc:', score[1])
    
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('bottleneck Training accuracy and loss')
    plt.legend()
    plt.savefig('bottleneck Training accuracy and loss.jpg')
    plt.figure()
    
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('bottleneck loss')
    plt.legend()
    plt.savefig('bottleneck Training loss and validation loss.jpg')
    plt.figure()
    
    plt.plot(epochs, val_acc, 'bo', label='Validation acc')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('bottleneck validation accuracy and loss')
    plt.legend()
    plt.savefig('bottleneck validation accuracy and loss.jpg')
    
    plt.show()
    model.save_weights('bottleneck_fc_model.h5')
    model.save('imagevgg.h5')

if __name__ == '__main__':
    train_fc()