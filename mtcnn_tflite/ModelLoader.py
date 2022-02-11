from mtcnn_tflite.ModelBuilder import ModelBuilder

import tensorflow as tf
import os
import mtcnn_tflite

class ModelLoader(ModelBuilder):
    def __init__(self, min_face_size=20, scale_factor=0.709):
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        data_path = os.path.join(os.path.dirname(mtcnn_tflite.__file__), "data")
        self.weights_file = os.path.join(data_path, "mtcnn_weights.npy")
        self.r_net = tf.lite.Interpreter(model_path=os.path.join(data_path, "r_net.tflite"))
        self.o_net = tf.lite.Interpreter(model_path=os.path.join(data_path, "o_net.tflite"))

