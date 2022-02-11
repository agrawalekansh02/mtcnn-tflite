from mtcnn_tflite.ModelBuilder import ModelBuilder

import tensorflow as tf
import os
import mtcnn_tflite

class ModelGenerator(ModelBuilder):
    def __init__(self, min_face_size=20, scale_factor=0.709):
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor

        data_path = os.path.join(os.path.dirname(mtcnn_tflite.__file__), "data")
        self.weights_file = os.path.join(data_path, "mtcnn_weights.npy")
        
        r_net = self.build_rnet()
        converter = tf.lite.TFLiteConverter.from_keras_model(r_net)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        r_net = converter.convert()
        with open(os.path.join(data_path, 'r_net.tflite'), 'wb') as f:
            f.write(r_net)
        print("saved r.net")

        o_net = self.build_onet()
        converter = tf.lite.TFLiteConverter.from_keras_model(o_net)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        o_net = converter.convert()
        with open(os.path.join(data_path, 'o_net.tflite'), 'wb') as f:
            f.write(o_net)
        print("saved o.net")