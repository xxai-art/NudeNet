#!/usr/bin/env python
import pillow_avif  # noqa
from PIL import Image, ImageFont, ImageDraw
import os
import cv2
import numpy as np
import onnxruntime

__labels = [
    "女生殖器",
    "女脸",
    "屁股裸露",
    "女胸裸露",
    "女生殖器裸露",
    "男胸裸露",
    "肛门裸露",
    "足裸露",
    "腹部",
    "足",
    "腋窝",
    "腋窝裸露",
    "男脸",
    "腹部裸露",
    "男生殖器裸露",
    "肛门",
    "女胸",
    "屁股",
]


def _read_image(img, input_width, input_height):
  # From ultralytics
  img_height, img_width = img.shape[:2]
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (input_width, input_height))
  image_data = np.array(img) / 255.0
  image_data = np.transpose(image_data, (2, 0, 1))
  image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
  return image_data, img_width, img_height


def _postprocess(output, img_width, img_height, input_width, input_height):
  outputs = np.transpose(np.squeeze(output[0]))
  rows = outputs.shape[0]
  boxes = []
  scores = []
  class_ids = []
  x_factor = img_width / input_width
  y_factor = img_height / input_height

  for i in range(rows):
    classes_scores = outputs[i][4:]
    max_score = np.amax(classes_scores)

    if max_score >= 0.5:
      class_id = np.argmax(classes_scores)
      x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
      left = int((x - w / 2) * x_factor)
      top = int((y - h / 2) * y_factor)
      width = int(w * x_factor)
      height = int(h * y_factor)
      class_ids.append(class_id)
      scores.append(max_score)
      boxes.append([left, top, width, height])

  indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)

  detections = []
  for i in indices:
    box = boxes[i]
    score = scores[i]
    class_id = class_ids[i]
    detections.append((class_id, float(score), box))

  return detections


class NudeDetector:

  def __init__(self):
    self.onnx_session = onnxruntime.InferenceSession(
        os.path.join(os.path.dirname(__file__), "best.onnx"))
    model_inputs = self.onnx_session.get_inputs()
    input_shape = model_inputs[0].shape
    self.input_width = input_shape[2]
    self.input_height = input_shape[3]
    self.input_name = model_inputs[0].name

  def detect(self, image_bin):
    preprocessed_image, image_width, image_height = _read_image(
        image_bin, self.input_width, self.input_height)
    outputs = self.onnx_session.run(None,
                                    {self.input_name: preprocessed_image})
    detections = _postprocess(outputs, image_width, image_height,
                              self.input_width, self.input_height)

    return detections


if __name__ == "__main__":
  from os.path import abspath, basename, dirname, join
  from glob import glob
  DIR = dirname(dirname(abspath(__file__)))

  detector = NudeDetector()

  font_path = join(DIR, 'font/DroidSansFallback.ttf')
  font = ImageFont.truetype(font_path, 20, encoding="utf-8")

  for fp in glob(join(DIR, 'img/*.avif')):
    print(fp)
    img = Image.open(fp)
    detections = detector.detect(np.array(img.convert('RGB')))
    canvas = ImageDraw.Draw(img)

    n = 0
    for (kind, score, box) in detections:
      name = __labels[kind]
      score = int(1000 * score) / 10
      print(name, score, box)
      p1 = box[:2]
      p2 = box[2:]
      n += 1
      color = [255, 255, 255]
      color[n % 3] = 0
      color = tuple(color)
      canvas.text([p1[0] + 5, p1[1] + 10],
                  f"{name} {score}",
                  color,
                  font=font,
                  stroke_width=1,
                  stroke_fill=(0, 0, 0))
      canvas.rectangle(xy=(p1[0], p1[1], p1[0] + p2[0], p1[1] + p2[1]),
                       fill=None,
                       outline=color,
                       width=2)

      # cv2.rectangle(img, p1, [], (0, 255, 0), 2)
      # cv2.putText(img, f'{name} {score}', [p1[0], p1[1] - 15], font, 1,
      #             (0, 255, 255), 2)

    name = basename(fp)[:-5]
    img.save(join(DIR, f'out/{name}.avif'), quality=80)
