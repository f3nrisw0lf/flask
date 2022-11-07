from flask import Flask, request, jsonify
from flask_limiter import Limiter
from waitress import serve
from flask_limiter.util import get_remote_address
from flask_expects_json import expects_json

import os
from utility import import_tensorflow
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per minute"],
    storage_uri="memory://",
)

tf = import_tensorflow()

rnn_model = tf.keras.models.load_model('saved_model/rnn/v1', compile=False)


def is_hate_speech_many(predictions):
    verdicts = []

    for i in predictions:
        if(i >= 0):
            verdicts.append(1)
        else:
            verdicts.append(0)

    return verdicts


def is_hate_speech(prediction):
    return prediction[0] >= 0


single_prediction_schema = {
    "type": "object",
    "properties":  {
        "text": {"type": "string"}
    },
    "required": ["text"],
    "additionalProperties": False
}


@app.route("/single-hate-prediction", methods=['POST'])
@expects_json(single_prediction_schema)
@limiter.limit("100 per minute")
def single_hate_prediction():
    data = request.get_json()
    text = data['text']

    if(not text):
        return jsonify(is_hate_speech=f"{False}")

    verdict = is_hate_speech_many(rnn_model.predict([text])[0])

    return jsonify(is_hate_speech=verdict[0])


many_prediction_schema = {
    "type": "object",
    "properties":  {
        "texts": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["texts"],
    "additionalProperties": False
}


@app.route('/many-hate-prediction', methods=['POST'])
@expects_json(many_prediction_schema)
@limiter.limit("100 per minute")
def many_hate_prediction():
    data = request.get_json()

    verdicts = is_hate_speech_many(rnn_model.predict(data['texts']))

    response = []
    for i, value in enumerate(verdicts):
        response.append(
            {f"{i}": {"is_hate": f"{value}", f"original": f"{data['texts'][i]}"}})

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
