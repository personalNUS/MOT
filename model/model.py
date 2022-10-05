from concurrent.futures import process
import cv2
import tensorflow as tf

IMG_DIM = (416, 416)
CONFIDENCE_THRESHOLD = 0.5

class Model:
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
   def drawBoundingBox(coords):
      print()

   @staticmethod
   def predictOnLocalImg(od_model, imgFilePath):
      img = cv2.imread(imgFilePath)
      inp_tensor = Model.processImg(img) 

      preds = od_model.predict(inp_tensor)[0]
      class_ids = []
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
            scores = row[5:]

