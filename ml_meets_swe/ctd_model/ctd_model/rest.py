from flask import Flask, request
from dask.distributed import Client
from dask import delayed
import numpy as np


client = Client("127.0.0.1:8786")


@delayed
def analyze(img):
    from ctd_model.model import predict
    return predict(img)


app = Flask(__name__)


@app.route("/", methods=["POST"])
def predict_img():

    img = np.array(request.json["img"])
    prediction = analyze(img)
    label, probability = client.compute(prediction).result()

    return "%s (%4.2f%%)"%(label, float(probability))


if __name__ == '__main__':
    app.run()