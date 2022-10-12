import cv2
import numpy as np
import tensorflow as tf

IMG_DIM = (416, 416)
CONFIDENCE_THRESHOLD = 0.5

class ObjectDetector:
   @staticmethod
   def loadModelFromYaml():
      with open('model/yolov5s.yaml') as yaml_file:
         model = tf.keras.models.model_from_yaml(yaml_file.read())
         print(model.summary())
      yaml_file.close()

      model.load_weights('model/weights/model_weights.pt')

      return model

   @staticmethod
   def processImg(img):
      # Process the image to send into the model
      inp = cv2.resize(img, IMG_DIM)
      inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
      inp_tensor = tf.convert_to_tensor(inp, dtype = tf.uint8)
      inp_tensor = tf.expand_dims(inp_tensor, 0)
      inp_tensor = inp_tensor / 255

      return inp_tensor

   @staticmethod
   def getBoundingBox(x_factor, y_factor, x, y, w, h):
      left = int((x - 0.5 * w) * x_factor)
      top = int((y - 0.5 * h) * y_factor)
      width = int(w * x_factor)
      height = int(h * y_factor)

      return np.array([left, top, width, height])

   @staticmethod
   def predictOnLocalImg(od_model, imgFilePath):
      img = cv2.imread(imgFilePath)
      inp_tensor = ObjectDetector.processImg(img) 

      preds = od_model.predict(inp_tensor)[0]
      confidences = []
      boxes = []

      rows = preds.shape[0]

      img_width, img_height, _ = img.shape
      x = img_width / IMG_DIM[0]
      y = img_height / IMG_DIM[1]

      for r in range(0, rows):
         row = preds[r]
         confidence = row[4]

         if confidence >= CONFIDENCE_THRESHOLD:
            confidences.append(confidence)
            boxes.append(ObjectDetector.getBoundingBox(x, y, row[0].item(), row[1].item(), row[2].item(), row[3].item()))           

      for i in range (0, len(boxes)):
         box = boxes[i]
         confidence = confidences[i]

         cv2.rectangle(img, box)
         cv2.rectangle(img, (box[0], box[1] - 20), (box[0] + box[2], box[1]))
         cv2.putText(img, f"Person (Confidence: {confidence})", (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))