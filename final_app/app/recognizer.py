import numpy as np
from PIL import Image

class DummyRecognizer:
    def encode(self, img: Image.Image) -> np.ndarray:
        return np.random.rand(128)
