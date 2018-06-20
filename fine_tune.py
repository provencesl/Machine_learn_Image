# -*- coding: utf-8 -*-

from keras.applications.vgg16 import VGG16   
from keras.layers import Activation, Dropout, Flatten,Dense, Input, Reshape
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.models import Model
import matplotlib.pyplot as plt


def fine_tune():

    model = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
    base_model = Model(input = model.input, output = model.get_layer("block5_pool").output)
    
    for layer in model.layers[0:14]: #冻结前14层
        layer.trainable = False
        

    base_out = base_model.output
    base_out = Reshape((25088,))(base_out) #将卷积层输出reshape成一维向量
    top_model = Dense(256, activation='relu')(base_out)
    top_model = Dropout(0.5)(top_model)
    top_preds = Dense(4, activation='softmax',kernel_regularizer = regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.001))(top_model)
    my_model = Model(input = base_model.input, output=top_preds)
   
    
    
        
    my_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,  #剪切强度
            zoom_range=0.2,   #随机缩放
            horizontal_flip=True)  #随机水平翻转
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
            'E:/TensorFlowfile/imageclassification/data/train',
            target_size=(224, 224),
            batch_size=20,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            'E:/TensorFlowfile/imageclassification/data/validation',
            target_size=(224, 224),
            batch_size=20,
            class_mode='categorical')
   
  
    hist = my_model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=50,
            validation_data=validation_generator
            )
    score = my_model.evaluate_generator(validation_generator, 80)
    print('test acc:', score[1])
    my_model.save_weights('tuned_weights.h5')
    my_model.save('tuned.h5')
    
    
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training accuracy and loss')
    plt.legend()
    plt.savefig('tuned Training acc and loss.jpg')
    plt.figure()
    
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('loss')
    plt.legend()
    plt.savefig('Training loss and validation loss.jpg')
    plt.figure()
    
    plt.plot(epochs, val_acc, 'bo', label='Validation acc')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('validation accuracy and loss')
    plt.legend()
    plt.savefig('tuned validation acc and loss.jpg')
    
    plt.show()
    
    
if __name__ == '__main__':
    fine_tune()