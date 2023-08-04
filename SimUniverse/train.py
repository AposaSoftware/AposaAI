from ultralytics import YOLO

model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    model.train(data='SimUniverse.yaml', epochs=20000, imgsz=640)