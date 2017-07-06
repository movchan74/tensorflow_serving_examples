import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.contrib.session_bundle import exporter
import keras.backend as K

K.set_learning_phase(0)

#create model
model = ResNet50(weights='imagenet')

sess = K.get_session()

export_path = './model'
export_version = 1

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)

model_exporter.init(sess.graph.as_graph_def(), named_graph_signatures={
  'inputs': exporter.generic_signature({'images': model.input}),
  'outputs': exporter.generic_signature({'scores': model.output})})

model_exporter.export(export_path, tf.constant(export_version), sess)