import imp
from detection.scrfd.demo.detect import inference as detect
from recognition.arcface_torch.recognize import inference as recognize
import time
import cv2 
image = cv2.imread('/AIHCM/AI_Member/thanglq/insightface/detection/scrfd/prime.jpg')
for i in range(10):
    start = time.time()
    detections = detect(image)
    for detection in detections:
        embedding = recognize(detection)
    print('Total time:',time.time()-start)
    