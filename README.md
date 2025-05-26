# 🎯 Real-Time Object Detection with YOLOv8

**End-to-end real-time object detection system using YOLOv8, PyTorch (Ultralytics), and COCO128 Dataset.**

---

## 🗂 Project Directory Structure
PythonProject/
├── data/
│   └── coco128/
│       ├── images/
│       ├── labels/
│       └── coco128.yaml
├── runs/
│   └── detect/
│       ├── train3/
│       └── predict2/
├── scripts/
│   └── explore_coco128.py
├── yolov8n.pt
├── RealTime-ObjectDetection-YOLO.py
└── README.md

---

## 🚀 Project Highlights

- **Dataset Exploration**: Visualized YOLO-formatted bounding boxes on sample images.
- **Model Training**: YOLOv8 trained successfully for 10 epochs.
- **Inference**: Performed prediction on a batch of images.

---

## 🔧 Technical Setup

### 1. **Dependencies**

Install required libraries:
```bash
pip install ultralytics opencv-python matplotlib


2. Dataset
	•	Dataset used: COCO128 Dataset
	•	A lightweight subset of COCO, perfect for rapid experimentation.

3. Dataset Exploration

Sample visualization of dataset with bounding boxes:
import os
import cv2
import matplotlib.pyplot as plt

img_dir = '/path_to_project/data/coco128/images/train2017'
label_dir = '/path_to_project/data/coco128/labels/train2017'

classes = [ ... ] # (Full class list here)

def draw_yolo_box(img, label_path):
    h, w = img.shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls, xc, yc, bw, bh = map(float, parts)
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cls = int(cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, classes[cls], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

for fname in os.listdir(img_dir)[:5]:
    img_path = os.path.join(img_dir, fname)
    label_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))
    img = cv2.imread(img_path)
    draw_yolo_box(img, label_path)
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(fname)
    plt.axis('off')
    plt.show()

4. Model Training

Run YOLO training:
yolo task=detect mode=train model=yolov8n.pt data=data/coco128/coco128.yaml epochs=10 imgsz=640

5. Model Prediction

Run inference on all images:
yolo task=detect mode=predict model=runs/detect/train3/weights/best.pt source=data/coco128/images/train2017/ show=True


💡 Lessons Learned
	•	Object Detection Pipeline (Training → Validation → Prediction)
	•	Dataset handling and YOLO annotations
	•	Visualization techniques for bounding box annotations

⚙️ Tools & Technologies
	•	Python 3
	•	PyTorch
	•	Ultralytics YOLOv8
	•	Matplotlib
	•	OpenCV






