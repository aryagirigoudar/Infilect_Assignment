import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_explain.core.activations import ExtractActivations
from tensorflow.keras.applications.xception import decode_predictions
from keras.applications.inception_v3 import InceptionV3

# Model InceptionV3
model = InceptionV3(weights='imagenet')

def run(path):

    # preprocess the image like scale down to 299X299
    img = tf.keras.preprocessing.image.load_img(path, target_size=(299,299))
    img = tf.keras.preprocessing.image.img_to_array(img)

    # make predictions
    img = tf.keras.applications.xception.preprocess_input(img)
    predictions = model.predict(np.array([img]))
    result = decode_predictions(predictions,top=1)[0][0]
    
    return result