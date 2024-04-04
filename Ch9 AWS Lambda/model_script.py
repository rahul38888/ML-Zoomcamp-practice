from io import BytesIO
import numpy as np
import tflite_runtime.interpreter as tflite
from urllib import request
from PIL import Image

classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def preprocess_url_image(url, size):
    img = download_image(url)
    img = img.resize(size, Image.NEAREST)
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X /= 127.5
    X -= 1.0
    return X

class ClothingTFLiteModel:
    def __init__(self, model_path, classes, input_image_size) -> None:
        self.classes = classes
        self.input_image_size = input_image_size
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def interpret(self, X):
        self.interpreter.set_tensor(self.input_index, X)
        self.interpreter.invoke()
        y_tf = self.interpreter.get_tensor(self.output_index)

        y_tf = y_tf[0].tolist()

        return dict(zip(self.classes, y_tf))
    
    def interpret_from_url(self, url):
        return self.interpret(preprocess_url_image(url, self.input_image_size))

colothing_model = ClothingTFLiteModel('clothing_xception_v4_23_0.911.tflite', classes, (299, 299))

# print(colothing_model.interpret_from_url('http://bit.ly/mlbookcamp-pants'))


def lambda_handler(event, context):
    url = event['url']
    result = colothing_model.interpret_from_url(url=url)

    return result