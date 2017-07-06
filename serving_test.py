# coding: utf-8

import time
import sys
import tensorflow as tf
import numpy as np
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_prediction(host, port, img_path):
    image = preprocess_image(img_path)

    start_time = time.time()
    
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet50'

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=image.shape))

    result = stub.Predict(request, 10.0)
    prediction = np.array(result.outputs['scores'].float_val)
    
    return prediction, (time.time()-start_time)*1000.

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print ('usage: serving_test.py <host> <port> <img_path>')
        print ('example: serving_test.py 127.0.0.1 9001 ~/elephant.jpg')
        exit()

    host = sys.argv[1]
    port = int(sys.argv[2])
    img_path = sys.argv[3]

    for i in range(10):
        prediction, elapsed_time = get_prediction(host, port, img_path)
        if i == 0:
            print('Predicted:', decode_predictions(np.atleast_2d(prediction), top=3)[0])
        print('Elapsed time:', elapsed_time, 'ms')