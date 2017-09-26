import json

from flask import Flask
from flask import request

from dask.distributed import Client
from dask import delayed

import numpy as np


@delayed
def analyze(img):
    from ctd_model.model import predict
    return predict(img)


client = Client("127.0.0.1:8786")

app = Flask(__name__)

@app.route("/", methods=["POST"])
def predict_img():
    payload = request.json

    img = np.array(payload["img"], dtype=np.int8)

    future = client.compute(analyze(img))

    label, prob = future.result()
    return json.dumps(dict(
        label=int(label),
        probability=float(prob)
        ))


if __name__ == '__main__':
    app.run()