import cv2

from mtcnn_tflite.exceptions import InvalidImage
from mtcnn_tflite.MTCNN import MTCNN

def main():
    mtcnn = MTCNN()
    mtcnn.generate()
    mtcnn.load()

if __name__ == '__main__':
    main()