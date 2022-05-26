import sys
sys.path.append("C:/Users/linhnh/Desktop/TMT/face_recognize/detection/retinaface")

import cv2 
import glob
import os
import json
import mxnet as mx
from tqdm import tqdm
from collections import namedtuple

from detection.retinaface import retinaface, face_inference
from recognition.arcface_mxnet.inference_face_embedding import get_face_embeded


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
        face_embeded = get_face_embeded(image, model_recognition)
        embedding[dir].append([face_embeded.tolist()])
        count += 1

with open(os.path.join(output_path, 'face_database.json'), 'w') as f:
    json.dump(embedding, f)
