import os
import cv2
import matplotlib.pyplot as plt

# Paths to images and labels (adjusted for your structure)
img_dir = '/Users/shiveshrajsahu/Desktop/PythonProject/data/coco128/images/train2017'
label_dir = '/Users/shiveshrajsahu/Desktop/PythonProject/data/coco128/labels/train2017'

# COCO class labels
classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

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

# Visualize 5 images
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


"""
Achieved So Far
Professional Project Structure: Folders organized, dataset in the right place.
Data Understanding: visualize and interpret the YOLO annotation format.
Hands-On Experience: working with real computer vision datasets just like in top AI industry roles.
	"""

"""
Step 1: Train YOLOv8 on coco128
Open terminal and run:
yolo task=detect mode=train model=yolov8n.pt data=data/coco128/coco128.yaml epochs=10 imgsz=640
What this does:
task=detect: Tells YOLO you want to do object detection.
mode=train: Training mode.
model=yolov8n.pt: Uses the small, fast YOLOv8n model for quick training.
data=data/coco128/coco128.yaml: Points to your dataset and classes.
epochs=10: 10 training passes 
imgsz=640: Standard image size for detection.
"""
"""
What Happens Next?
Training will run for 10 epochs.
You’ll see metrics like loss and mAP (mean average precision) being updated as the model learns.
Results, charts, and model weights are being saved in a folder (typically runs/detect/train/).
"""
"""
After Training Finishes:

1. evaluate model performance.
YOLOv8 will automatically evaluate on the validation set after the last epoch.
see metrics for accuracy, precision, recall, etc.

2. run inference on images or webcam.
Test the trained model (best.pt) on new images or webcam!
Example command (in terminal, after training finishes):
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=
...'data/coco128/images/val2017/000000000139.jpg' show=True

or to use webcam:
yolo task=detect mode=predict model=
...runs/detect/train/weights/best.pt source=0 show=True
"""
"""
Proved and Learned:
Data scientist/ML engineer workflow: dataset handling, annotations, visual verification.
Model training and experiment management: running, debugging, and tracking model progress.
Professionalism: Project structure is neat, reproducible.
"""

"""
What to do next:
Let the model finish training.
Try out the inference command above on any image or webcam.

"""
"""
Pro Tip:
want to predict on all images in that folder, run:
yolo task=detect mode=predict model=runs/detect/train3/weights/best.pt source='...coco128/images/train2017/' show=True
"""



