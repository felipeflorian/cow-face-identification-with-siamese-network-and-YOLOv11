from ultralytics import YOLO

model_11 = YOLO("best.pt")

def get_yolo_coordinates(path, model):

  results = model(path, save_txt=False, save=False, imgsz=640, conf=0.5)
  vals = results[0].boxes.xywh
  vals = vals.cpu()
  x_center, y_center, width, height = vals.numpy().flatten()

  return x_center, y_center, width, height