import sys

sys.path.append("C:/Users/linhnh/Desktop/TMT/insightface/detection/scrfd")

import torch
from detection.scrfd.demo.detect import inference as detect
from recognition.arcface_torch.recognize import inference as recognize
import time
import cv2 
import glob
import os, random
import json
from tqdm import tqdm

from detection.scrfd.mmdet.apis import init_detector
from recognition.arcface_torch.backbones import get_model

start = time.time()

model_detection = init_detector("./scrfd_500m_bnkps.py", "model_detect.pth", device='cpu') 

model_recognition = get_model(name='r100', fp16=False)
model_recognition.load_state_dict(torch.load("backbone_r100.pth", map_location="cpu"))
model_recognition.eval()


dataset_path = 'C:/Users/linhnh/Desktop/TMT/insightface/dataset/'
embedding = {}
for dir in tqdm(os.listdir(dataset_path)):
    embedding[dir] = []
    for img_file in tqdm(glob.glob(os.path.join(dataset_path, dir, '*.jpg'))):
        image = cv2.imread(img_file)
        detections = detect(model_detection, image)
        for detection in detections:
            cv2.imwrite("test_{}.jpg".format(random.randint(0, 100)), detection)
            vector = recognize(model_recognition, detection)
            embedding[dir].append(vector.tolist())

db_path = 'C:/Users/linhnh/Desktop/TMT/insightface/'
with open(os.path.join(db_path, 'db.json'), 'w') as f:
    json.dump(embedding, f)
    
print('Time embedding: ', time.time() - start)
