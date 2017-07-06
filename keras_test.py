# coding: utf-8

import sys
import time
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_prediction(model, img_path):
    image = preprocess_image(img_path)

    start_time = time.time()
    prediction = model.predict(image)

    return prediction, (time.time()-start_time)*1000.



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print ('usage: keras_test.py <img_path>')
        print ('example: keras_test.py ~/elephant.jpg')
        exit()

    img_path = sys.argv[1]

    model = ResNet50(weights='imagenet')

    for i in range(10):
        prediction, elapsed_time = get_prediction(model, img_path)
        if i == 0:
            print('Predicted:', decode_predictions(np.atleast_2d(prediction), top=3)[0])
        print('Elapsed time:', elapsed_time, 'ms')