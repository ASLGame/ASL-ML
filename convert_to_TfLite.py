import tensorflow as tf

model = tf.keras.models.load_model('model/R_keypoint_classifier_final.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("tflite_models/R_converted_model.tflite", "wb").write(tflite_model)

model = tf.keras.models.load_model('model/L_keypoint_classifier_final.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("tflite_models/L_converted_model.tflite", "wb").write(tflite_model)