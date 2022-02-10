# MTCNN face recognition

Implementation of the [MTCNN face detection algorithm](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7553523). This project converted the code from [ipazc/mtcnn](https://github.com/ipazc/mtcnn) to TF Lite. My iteration takes the code a little further by enabling a feature to download and load a saved `tflite` model. This enable the ability to use `MTCNN` in a mobile environment.

## Installation

You can install the package through pip:

```
pip install mtcnn-tflite
```

## Quick start

Similar to [the original implementation](https://github.com/ipazc/mtcnn), the following example illustrates the ease of use of this package:

```
>>> from mtcnn_tflite.MTCNN import MTCNN
>>> import cv2
>>>
>>> img = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
>>> detector = MTCNN()
>>> detector.detect_faces(img)
[
    {
        'box': [276, 88, 51, 68],
        'confidence': 0.9989245533943176,
        'keypoints': {
            'left_eye': (291, 117),
            'right_eye': (314, 114),
            'nose': (303, 130),
            'mouth_left': (296, 143),
            'mouth_right': (314, 141)
        }
    }
]
```


## Benchmark

| Image size | TF version                            | Process time * |
|------------|---------------------------------------|----------------|
| 561x561    | [TF2](https://github.com/ipazc/mtcnn) | 698ms          |
| 561x561    | **This repository** (TF Lite)         | 445ms          |

\* executed on a CPU: Intel i7-10510U

## License

[MIT License](https://github.com/mobilesec/mtcnn-tflite/blob/master/LICENSE)

## Acknowledgement
This work has been carried out within the scope of Digidow, the Christian Doppler Laboratory for Private Digital Authentication in the Physical World, funded by the Christian Doppler Forschungsgesellschaft, 3 Banken IT GmbH, Kepler Universitätsklinikum GmbH, NXP Semiconductors Austria GmbH, and Österreichische Staatsdruckerei GmbH and has partially been supported by the LIT Secure and Correct Systems Lab funded by the State of Upper Austria.

