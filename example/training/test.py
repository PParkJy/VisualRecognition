from IPython.display import display
from PIL import Image
from yolo import YOLO

def objectDetection(file, model_path, class_path):
    yolo = YOLO(model_path=model_path, classes_path=class_path, anchors_path='model_data/tiny_yolo_anchors.txt')
    image = Image.open(file)
    result_image = yolo.detect_image(image)
    display(result_image)

objectDetection('../dataset/test/IMG_20181228_102636.jpg', 'logs/000/trained_weights_final.h5', 'model_data/class.txt')