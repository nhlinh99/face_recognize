import sys
sys.path.append("C:/Users/linhnh/Desktop/TMT/face_recognize/detection/scrfd")

import cv2
import time
import json
import torch
import faiss
# import annoy
import numpy as np
import pyrealsense2 as rs
from sklearn import preprocessing

from detection.scrfd.mmdet.apis import init_detector
from recognition.arcface_torch.backbones import get_model
from detection.scrfd.demo.detect import inference as detect
from recognition.arcface_torch.recognize import inference as recognize

EMBEDDING_DIMENSION = 512

def load_db(db_path, use_gpu = False):
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
            list_feature.append(np.array(feature).astype('float32').reshape(1,EMBEDDING_DIMENSION))

    list_feature = np.concatenate(list_feature , axis=0)
    list_feature_new = preprocessing.normalize(list_feature, norm='l2')
    index.add(list_feature_new)
    
    return list_len, list_id, index


def identification(image, model_detection, model_recognition, threshold_detect, threshold_recog):
    list_len_embedding, list_person_name, index_faiss = face_database
    result = []
    face_info = detect(model_detection, image, threshold_detect)
    for i in range(len(face_info)):
        face = face_info[i][0]
        bounding_box = face_info[i][1]
        face_embeded = recognize(model_recognition, face)[0]
        xq = face_embeded.astype('float32').reshape(1, EMBEDDING_DIMENSION)
        xq = preprocessing.normalize(xq, norm='l2')
        distances, indices = index_faiss.search(xq, 1)
        
        position = indices[0][0]
        sum = 0
        for idx in range(len(list_person_name)):
            sum += list_len_embedding[idx]
            if position < sum:
                if distances[0][0] >= threshold_recog:
                    result.append([list_person_name[idx], bounding_box, distances[0][0]])
                else:
                    result.append(["stranger", bounding_box, distances[0][0]])
                break
    return result


face_database = load_db("./models/face_database_torch.json")

model_detection = init_detector("scrfd_500m_bnkps.py", "models/detection/model_detect.pth", device='cpu') # cuda:0 

# define mxnet
model_recognition = get_model(name='r34', fp16=False)
model_recognition.load_state_dict(torch.load("models/recognition/backbone_r34.pth", map_location="cpu"))
model_recognition.eval()

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

padding = 0

prev_frame_time = 0
new_frame_time = 0

while True:
    frames = pipeline.wait_for_frames(1000)

    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    color_image = np.array(color_frame.get_data())

    list_person_info = identification(color_image, model_detection, model_recognition, threshold_detect=0.5, threshold_recog=0.7)
    for person_info in list_person_info:
        name = person_info[0]
        bounding_box = person_info[1]
        distance = round(person_info[2], 2)
        cv2.rectangle(color_image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)
        cv2.putText(color_image, "{} {}".format(name, str(distance)), (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)
    fps = "FPS: " + str(fps)

    # putting the FPS count on the frame
    cv2.putText(color_image, fps, (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    cv2.namedWindow("Align Example", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Align Example", color_image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key ==27:
        cv2.destroyAllWindows()
        break

# color_image = cv2.imread("test.png")
# list_person_info = identification(color_image, model_detection, model_recognition, threshold_detect=0.5, threshold_recog=0.7)
# for person_info in list_person_info:
#     name = person_info[0]
#     bounding_box = person_info[1]
#     distance = round(person_info[2], 2)
#     cv2.rectangle(color_image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)
#     cv2.putText(color_image, "{} {}".format(name, str(distance)), (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# cv2.imwrite("res.jpg", color_image)
