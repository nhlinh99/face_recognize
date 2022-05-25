import argparse
import sys
import cv2
import numpy as np
import torch
import time


def inference(net,img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    with torch.no_grad():
        feat = net(img).numpy()

    return feat
# if __name__ == "__main__":
#     start = time.time()
#     for i in range(5):
#         inference(img)
#     print('totaltime',time.time()-start)