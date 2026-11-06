'''
Object detection in 10 lines of code with the imageai library.

- Based on the article:
  https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606
- Library: https://github.com/OlafenwaMoses/ImageAI
- Install:
  pip install tensorflow numpy scipy opencv-python pillow matplotlib h5py imageai
- Download the RetinaNet model file (resnet50_coco_best_v2.0.1.h5) from:
  https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
  and set the MODEL_AI_PATH environment variable to the folder containing it.
- "image.jpg" is the source file and "imagenew.jpg" is the output file
- picture from thai Drama: "เลือดข้นคนจาง"
'''
import os

from imageai.Detection import ObjectDetection
from PIL import Image

# folder that contains resnet50_coco_best_v2.0.1.h5 (default: this folder)
h5_path = os.environ.get("MODEL_AI_PATH", ".")

if __name__ == "__main__":
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(h5_path, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    execution_path = ""
    detections = detector.detectObjectsFromImage(
        input_image=os.path.join(execution_path, "image.jpg"),
        output_image_path=os.path.join(execution_path, "imagenew.jpg"))
    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"])

    image = Image.open('imagenew.jpg')
    image.show()
