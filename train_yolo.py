from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='yolo_dataset/dataset.yaml',
    epochs=15,
    imgsz=512,
    batch=16,
    name='dji_detector',
    project='checkpoints/yolo',
    patience=5,
    device='cpu',
    workers=0,
    pretrained=True,
    optimizer='Adam',
    lr0=0.001,
    augment=True,
    fliplr=0.5,
    translate=0.2,
    scale=0.3,
    degrees=15.0,
    mosaic=1.0
)
print('Training complete')
