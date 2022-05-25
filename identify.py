import cv2
import time
import json
import faiss
import torch
import numpy as np
import pyrealsense2 as rs
from sklearn import preprocessing

from detection.scrfd.mmdet.apis import init_detector
from recognition.arcface_torch.backbones import get_model
from detection.scrfd.demo.detect import inference as detect
from recognition.arcface_torch.recognize import inference as recognize


def load_db(db_path, use_gpu = True):
    with open(db_path, 'r') as f:
        db = json.load(f)
        
    first_time = True
    list_feature = []
    list_id = []
    list_len = []
    for k,v in db.items():
        list_id.append(k)
        list_len.append(len(v))
        if first_time:
            d = np.array(v[0]).shape[1]
            index = faiss.IndexFlatIP(d)
            if use_gpu:
                device =  faiss.StandardGpuResources()  # use a single GPU
                index = faiss.index_cpu_to_gpu(device, 0, index)
            first_time = False
        for feature in v:
            list_feature.append(np.array(feature).astype('float32').reshape(1,512))

    list_feature = np.concatenate(list_feature , axis=0)
    list_feature_new = preprocessing.normalize(list_feature, norm='l2')
    index.add(list_feature_new)
    
    return list_len, list_id, index


def identification(image, model_detection, model_recognition, threshold):
    list_len_embedding, list_person_name, index_faiss = face_database
    result = []
    detections = detect(model_detection, image)
    
    for detection in detections:
        croped_image = detection[0]
        bounding_box = detection[1]
        face_embedding = recognize(model_recognition, croped_image)[0]
        
        xq = face_embedding.astype('float32').reshape(1,512)
        xq = preprocessing.normalize(xq, norm='l2')
        distances, indices = index_faiss.search(xq, 1)
        
        position = indices[0][0]
        sum = 0
        for idx in range(len(list_person_name)):
            sum += list_len_embedding[idx]
            if position < sum:
                if distances[0][0] >= threshold:
                    result.append([list_person_name[idx], bounding_box])
                    
                break
        
    return result


face_database = load_db("db.json", use_gpu=False)

model_detection = init_detector("./scrfd_500m_bnkps.py", "model_detect.pth", device='cpu') 

model_recognition = get_model(name='r100', fp16=False)
model_recognition.load_state_dict(torch.load("backbone_r100.pth", map_location="cpu"))
model_recognition.eval()

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

padding = 0

while True:
    frames =pipeline.wait_for_frames(1000)

    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    color_image = np.array(color_frame.get_data())

    list_person_info = identification(color_image, model_detection, model_recognition, threshold=0.4)

    for person_info in list_person_info:
        name = person_info[0]
        bounding_box = person_info[1]
        print(bounding_box)
        cv2.rectangle(color_image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)
        cv2.putText(color_image, name, (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.namedWindow("Align Example", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Align Example", color_image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key ==27:
        cv2.destroyAllWindows()
        break
