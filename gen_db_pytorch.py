import sys
sys.path.append("C:/Users/linhnh/Desktop/TMT/face_recognize/detection/scrfd")

import cv2 
import glob
import os
import json
import torch
from tqdm import tqdm

from detection.scrfd.mmdet.apis import init_detector
from recognition.arcface_torch.backbones import get_model
from recognition.arcface_torch.recognize import inference as recognize


# model_detection = init_detector("scrfd_500m_bnkps.py", "models/detection/model_detect.pth", device='cpu') # cuda:0 

# define mxnet
model_recognition = get_model(name='r34', fp16=False)
model_recognition.load_state_dict(torch.load("models/recognition/backbone_r34.pth", map_location="cpu"))
model_recognition.eval()

ext = ['png', 'jpg', 'jpeg']

dataset_path = "C:/Users/linhnh/Desktop/TMT/dataset/face"
output_path = "C:/Users/linhnh/Desktop/TMT/dataset/face_crop"
embedding = {}
for dir in os.listdir(dataset_path):
    print("Solving:", dir)
    if not os.path.isdir(os.path.join(output_path, dir)):
        os.mkdir(os.path.join(output_path, dir))

    embedding[dir] = []
    count = 0
    list_image_dir = []
    [list_image_dir.extend(glob.glob(os.path.join(output_path, dir, '*.' + e))) for e in ext]
    for img_file in tqdm(list_image_dir):
        image = cv2.imread(img_file)
        # crop_faces, faces, landmarks = face_inference.get_face_area(image, model_detection, 0.6, [480, 640])
        # for face in crop_faces:
            # cv2.imwrite(os.path.join(output_path, dir, "{}.jpg".format(count)), face)
        face_embeded = recognize(model_recognition, image)[0]
        embedding[dir].append([face_embeded.tolist()])
        count += 1

with open(os.path.join("C:/Users/linhnh/Desktop/TMT/face_recognize/models", 'face_database_torch.json'), 'w') as f:
    json.dump(embedding, f)
