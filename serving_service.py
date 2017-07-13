from flask import Flask
from flask import request
from flask import jsonify
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

application = Flask(__name__)

host = '127.0.0.1'
port = 9001

def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_prediction(img):
    image = preprocess_image(img)
    
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet50'
    request.model_spec.signature_name = 'predict'

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=image.shape))

    result = stub.Predict(request, 10.0)
    prediction = np.array(result.outputs['scores'].float_val)
    
    return decode_predictions(np.atleast_2d(prediction), top=3)[0]

@application.route('/predict', methods=['POST'])
def predict():
    if request.files.get('data'):
        img = request.files['data']
        resp = get_prediction(img)
        response = jsonify(resp)
        return response
    else:
        return jsonify({'status': 'error'})

if __name__ == "__main__":
    application.run()