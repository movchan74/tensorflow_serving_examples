import os
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras.backend as K

K.set_learning_phase(0)

#create model
model = ResNet50(weights='imagenet')

sess = K.get_session()

export_path_base = './model'
export_version = 1

export_path = os.path.join(
  tf.compat.as_bytes(export_path_base),
  tf.compat.as_bytes(str(export_version)))
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

model_input = tf.saved_model.utils.build_tensor_info(model.input)
model_output = tf.saved_model.utils.build_tensor_info(model.output)

prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'images': model_input},
      outputs={'scores': model_output},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
  sess, [tf.saved_model.tag_constants.SERVING],
  signature_def_map={
      'predict':
          prediction_signature,
  },
  legacy_init_op=legacy_init_op)

builder.save()

print('Done exporting!')