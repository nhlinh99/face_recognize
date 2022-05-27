import sys
sys.path.append("C:/Users/linhnh/Desktop/TMT/face_recognize/Pytorch_Retinaface")

import cv2
import time
import json
import torch
import faiss
# import annoy
import numpy as np
import pyrealsense2 as rs
from sklearn import preprocessing

from Pytorch_Retinaface.detect import inference
from Pytorch_Retinaface.data import cfg_mnet
from Pytorch_Retinaface.models.retinaface import RetinaFace
from recognition.arcface_torch.backbones import get_model
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

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def identification(image, model_detection, model_recognition, threshold_detect, threshold_recog):
    list_len_embedding, list_person_name, index_faiss = face_database
    result = []
    box_and_landmarks, face_crop = inference(model_detection, image, threshold_detect, cfg_mnet, device)
    for i in range(len(box_and_landmarks)):
        bounding_box = list(map(int, box_and_landmarks[i][0:4]))
        landmark_points = list(map(int, box_and_landmarks[i][5:15]))
        face = face_crop[i]
        landmarks = [[landmark_points[k], landmark_points[k+1]] for k in range(0, 10, 2)]

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
                    result.append([list_person_name[idx], bounding_box, landmarks, distances[0][0]])
                else:
                    result.append(["stranger", bounding_box, landmarks, distances[0][0]])
                break
    return result


face_database = load_db("./models/face_database_torch.json")

device = torch.device("cpu")

# Detection
model_detection = RetinaFace(cfg=cfg_mnet, phase = 'test')
model_detection = load_model(model_detection, "models/detection/mobilenet0.25_Final.pth", True)
model_detection.eval()
model_detection = model_detection.to(device)

# Recognition
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
        landmarks = person_info[2]
        distance = round(person_info[3], 2)
        [cv2.circle(color_image, (point[0], point[1]), 1, (255,0,0), 2) for point in landmarks]
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
#     landmarks = person_info[2]
#     distance = round(person_info[3], 2)
#     [cv2.circle(color_image, (point[0], point[1]), 1, (255,0,0), 2) for point in landmarks]
#     cv2.rectangle(color_image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)
#     cv2.putText(color_image, "{} {}".format(name, str(distance)), (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# cv2.imwrite("res.jpg", color_image)
