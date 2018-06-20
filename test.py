# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:20:05 2018

@author: Beobachter
"""
  
from keras.models import load_model  
import numpy as np  
import os 
import matplotlib.pyplot as plt
import matplotlib.image as pimage
from PIL import Image 
  
  
model = load_model('tuned.h5')  
#model.summary()




def test_one_image(test_img):
    image = test_img.resize([224, 224])
    image = np.array(image)
    im_array = image[np.newaxis, :]
    pre = model.predict(im_array)
    #print(pre)
    matt = np.argmax(pre)
    #print(matt)

    if matt == 0:
        print('This is a cantaloupe')
    if matt == 1:
        print('This is a cat')
    if matt == 2:
        print('This is a dog')
    if matt == 3:
        print('This is a watermelon')

test_dir = ['cantaloupe', 'cat', 'dog', 'watermelon']

for i in range(len(test_dir)):
    img_dir = 'E:/TensorFlowfile/imageclassification/data/test/' + test_dir[i] + '/'
    pic = os.listdir(img_dir)   
    n = len(pic)
    ind = np.random.randint(0, n)
    img = img_dir + pic[ind]
    image = pimage.imread(img) #float(32)
    plt.imshow(image)
    plt.show()
    image = Image.fromarray(np.uint8(image*255))  #uinit8
    test_one_image(image)
