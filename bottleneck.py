# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:21:13 2018

@author: Beobachter
"""


#导入预训练权重和网络框架
from keras.applications.vgg16 import VGG16 
#WEIGHTS_PATH = 'E:\TensorFlowfile\Kerasmodel\vgg16_weights_tf_dim_ordering_tf_kernels.h5'
#WEIGHTS_PATH_NO_TOP = 'E:\TensorFlowfile\Kerasmodel\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = VGG16(weights='imagenet',include_top=False)

#提取bottleneck特征

#载入图片
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def bottleneck():

    datagen = ImageDataGenerator(rescale = 1./255)#将每个像素缩放至[0,1]
    
    #训练集图像生成器
    train_generator = datagen.flow_from_directory(
            'E:/TensorFlowfile/imageclassification/data/train',
            target_size = (250,250),
            batch_size = 20,
            class_mode = None,
            shuffle = False
            )
    print('train begin')
    
    #验证集图像生成器
    validation_generator = datagen.flow_from_directory(
            'E:/TensorFlowfile/imageclassification/data/validation',
            target_size = (250,250),
            batch_size = 20,
            class_mode = None,
            shuffle = False
            )
    
    #灌入权重
    #model.load_weights(WEIGHTS_PATH_NO_TOP)
    
    #bottleneck feature
    bottleneck_feature_train = model.predict_generator(train_generator,100)
    np.save(open('bottleneck_feature_train.npy', 'wb+'), bottleneck_feature_train)
    bottleneck_feature_validation = model.predict_generator(validation_generator, 40)
    np.save(open('bottleneck_feature_validation.npy', 'wb+'), bottleneck_feature_validation)
    print('train done')

if __name__ == '__main__':
    bottleneck()
