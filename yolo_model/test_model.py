from ultralytics import YOLO
import cv2

model_11 = YOLO("best.pt")

def get_yolo_coordinates(path, model, img_size=(640, 640)):

  results = model(path, save_txt=False, save=False, imgsz=640, conf=0.5)
  vals = results[0].boxes.xywh
  vals = vals.cpu()
  x_center, y_center, width, height = vals.numpy().flatten()
  
  img = cv2.imread(path)
  
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  x1 = int(x_center - width / 2)
  y1 = int(y_center - height / 2)
  x2 = int(x_center + width / 2)
  y2 = int(y_center + height / 2)
  
  cropped_bbox = img_rgb[y1:y2, x1:x2]
  cow_face = cv2.resize(cropped_bbox, img_size, interpolation=cv2.INTER_AREA)

  return cow_face