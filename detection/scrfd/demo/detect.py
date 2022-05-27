import torch
from mmdet.apis import inference_detector, init_detector
import numpy as np
# config  = '/AIHCM/AI_Member/thanglq/insightface/detection/scrfd/configs/scrfd/scrfd_500m_bnkps.py'
# checkpoint = '/AIHCM/AI_Member/thanglq/insightface/detection/scrfd/model/model_detect.pth'
# device = 'cuda:0'
# score_thr = 0.5
# model = init_detector(config, checkpoint, device)

def inference(model, image, threshold=0.4):
    detections = []
    with torch.no_grad():
        result = inference_detector(model, image)
        for i in range(len(result[0])):
            bbox_temp =  torch.tensor(result[0][i]).view(1,5).tolist()[0]
            if bbox_temp[4] > threshold:
                bounding_box = np.array(bbox_temp[0:4]).astype(int).tolist()
                detections.append([image[bounding_box[1]: bounding_box[3], bounding_box[0]: bounding_box[2]],
                                    bounding_box])
            else:
                break
            
    return detections


# if __name__ == '__main__':
#     image = cv2.imread('prime.jpg')
#     for i in range(10):
#         inference(image)

