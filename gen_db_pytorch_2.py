import sys
sys.path.append("C:/Users/linhnh/Desktop/TMT/face_recognize/detection/scrfd")

import cv2 
import glob
import os
import json
import torch
from tqdm import tqdm
from PIL import Image

from MobileFaceNet.data_gen import data_transforms
from detection.scrfd.mmdet.apis import init_detector

def get_image(img, transformer):
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img.to(device)

device = torch.device("cpu")

checkpoint = 'models/recognition/mobilefacenet.pt'
print('loading model: {}...'.format(checkpoint))
model_recognition = torch.jit.load(checkpoint)
model_recognition.eval()
transformer = data_transforms['val']

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
        imgs = torch.zeros([1, 3, 112, 112], dtype=torch.float)
        imgs[0] = get_image(image, transformer)
        face_embeded = model_recognition(imgs.to(device)).cpu().detach().numpy()
        embedding[dir].append(face_embeded.tolist())
        count += 1

with open(os.path.join("C:/Users/linhnh/Desktop/TMT/face_recognize/models", 'face_database_128d.json'), 'w') as f:
    json.dump(embedding, f)
