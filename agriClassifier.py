

import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from keras.models import load_model
model = load_model('agriDrone3.h5')


def agriClass(img_path):
    class_names=['corn', 'jute', 'rice']
    dim=(600,600)
    frame=cv2.imread(img_path)
    
    frame=cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     
    #img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
    #img_array = keras.preprocessing.image.img_to_array(img)

    img_array = tf.expand_dims(frame, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    ph=0
    N=0
    K=0
    C=0
    P=0
    
    phLevels=[6.5, 4.8,5.5]
    NLevels=[25, 900, 2000]
    KLevels=[140, 80 , 40]
    CLevels=[40, 9.5, 60]
    PLevels=[300, 13, 400]
    
    for a,i in enumerate(score):
        ph= ph+(phLevels[a]*i)
        N= N+(NLevels[a]*i)
        K= K+ (KLevels[a]*i)
        C= C+ (CLevels[a]*i)
        P= P+ (PLevels[a]*i)
    params=dict()
    params["PH"]= "{:.1f}".format(ph)
    params["N"]= "{:.0f}".format(N)
    params["P"]= "{:.0f}".format(P)
    params["K"]= "{:.0f}".format(K)
    params["C"]= "{:.0f}".format(C)
    return class_names[np.argmax(score)], 100 * np.max(score), params
if __name__=="__main__":
    img_path="static//test2.jpg"
    print (agriClass(img_path))
