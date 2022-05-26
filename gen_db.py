import sys
sys.path.append("C:/Users/linhnh/Desktop/TMT/face_recognize/detection/retinaface")

import time
import cv2 
import glob
import os, random
import json
import mxnet as mx
from tqdm import tqdm
from collections import namedtuple

# from detection.scrfd.demo.detect import inference as detect
# from recognition.arcface_torch.recognize import inference as recognize
# from detection.scrfd.mmdet.apis import init_detector
# from recognition.arcface_torch.backbones import get_model

from detection.retinaface import retinaface, face_inference
from recognition.arcface_mxnet.inference_face_embedding import get_face_embeded

start = time.time()

model_detection = retinaface.RetinaFace('./models/detection/Mobilenet/mnet.25', 0, -1, 'net3')

prefix = "./models/recognition/MFaceNet/model"
sym, arg, aux = mx.model.load_checkpoint(prefix, 0)

# define mxnet
ctx = mx.cpu() # gpu_id = 0 ctx = mx.gpu(gpu_id)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
mod.set_params(arg, aux)
batch = namedtuple('Batch', ['data'])
model_recognition = [mod, batch]

dataset_path = 'C:/Users/linhnh/Desktop/TMT/face_recognize/dataset'
embedding = {}
for dir in tqdm(os.listdir(dataset_path)):
    embedding[dir] = []
    for img_file in tqdm(glob.glob(os.path.join(dataset_path, dir, '*.jpg'))):
        image = cv2.imread(img_file)
        crop_faces, faces, landmarks = face_inference.get_face_area(image, model_detection, 0.6, [480, 640])
        for face in crop_faces:
            # cv2.imwrite("test_{}.jpg".format(random.randint(0, 100)), face)
            face_embeded = get_face_embeded(face, model_recognition)
            embedding[dir].append([face_embeded.tolist()])

db_path = 'C:/Users/linhnh/Desktop/TMT/face_recognize'
with open(os.path.join(db_path, 'db.json'), 'w') as f:
    json.dump(embedding, f)
    
print('Time embedding: ', time.time() - start)
