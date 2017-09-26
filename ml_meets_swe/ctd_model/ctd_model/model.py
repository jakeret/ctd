from pkg_resources import resource_filename
import tensorflow as tf
from keras.models import load_model
import numpy as np
import ctd_model

graph = tf.get_default_graph()
inference_model = None

MODEL_NAME = "model.h5"

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def preprocess(img):
    img = img[np.newaxis, ...].astype(np.int8)
    return img


def predict(img):

    with graph.as_default():
        pred = inference_model.predict(preprocess(img))
        idx = np.argmax(pred)
        return (labels[int(idx)], pred[0, idx])


def dask_setup(service):

    print("Loading model")

    model_path = resource_filename(ctd_model.__name__, MODEL_NAME)
    print("Model path '%s'"%model_path)

    global inference_model
    inference_model = load_model(model_path)

    print("Model loaded")