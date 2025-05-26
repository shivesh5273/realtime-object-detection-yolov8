from ultralytics.data.dataset import YOLODataset
import matplotlib.pyplot as plt
import random
import cv2
import os

# Load dataset (coco128)
dataset = YOLODataset('datasets/coco128/images/train2017', task='detect')

# Randomly visualize 5 images
for i in random.sample(range(len(dataset)), 5):
    img, labels = dataset[i]
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Image {i} with annotations')
    plt.axis('off')
    plt.show()