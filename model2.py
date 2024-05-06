import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw
import os

class TensorHub:

    def __init__(self):
        
        self.MODEL_PATH = 'https://www.kaggle.com/models/tensorflow/efficientdet/tensorFlow2/d0/1'
        self.detector = hub.load(self.MODEL_PATH)

    def detect_persons(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path)
        image_tensor = tf.convert_to_tensor(np.array(image)[tf.newaxis, ...])
        
        # Perform object detection
        results = self.detector(image_tensor)

        # Draw bounding boxes around persons
        annotated_image = self.draw_bboxes(image, results)

        # Count the number of persons detected
        count = self.count_persons(results)

        # Save the annotated image
        annotated_image_path = os.path.join('Project','annotated_image.jpg')
        annotated_image.save(annotated_image_path)

        return count, annotated_image_path

    def draw_bboxes(self, image, data):
        draw = ImageDraw.Draw(image)

        im_width, im_height = image.size
        boxes = data['detection_boxes'].numpy()[0]
        classes = data['detection_classes'].numpy()[0]
        scores = data['detection_scores'].numpy()[0]

        for i in range(int(data['num_detections'][0])):
            if classes[i] == 1 and scores[i] > 0.25:  # Class ID 1 corresponds to persons
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
                draw.rectangle([left, top, right, bottom], outline="red", width=3)

        return image

    def count_persons(self, data):
        # Class ID 1 corresponds to persons
        count = sum(1 for score in data['detection_scores'].numpy()[0] if score > 0.25)
        return count-2