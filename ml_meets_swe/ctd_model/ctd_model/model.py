from pkg_resources import resource_filename
import tensorflow as tf
from keras.models import load_model
import numpy as np
import ctd_model

graph = tf.get_default_graph()
inference_model = None

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def dask_setup(service=None):
    model_path = resource_filename(ctd_model.__name__, "model.h5")

    global inference_model
    inference_model = load_model(model_path)

    print("Model loaded")


def predict(img):
    #preprocess
    img = img[np.newaxis, ...].astype('float32') / 255

    with graph.as_default():
        prediction = inference_model.predict(img)
        idx = np.argmax(prediction)

        return labels[int(idx)], float(prediction[0, idx])
