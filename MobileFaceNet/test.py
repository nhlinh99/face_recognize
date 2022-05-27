import time
import cv2
from PIL import Image
import torch
from data_gen import data_transforms

device = torch.device('cpu')
checkpoint = 'C:/Users/linhnh/Desktop/TMT/face_recognize/models/recognition/mobilefacenet.pt'
print('loading model: {}...'.format(checkpoint))
model = torch.jit.load(checkpoint)
# model = checkpoint['model'].to(device)
model.eval()
transformer = data_transforms['val']

def get_image(img, transformer):
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img.to(device)


filepath = "C:/Users/linhnh/Desktop/TMT/face_recognize/1.jpg"
imgs = torch.zeros([1, 3, 112, 112], dtype=torch.float)
imgs[0] = get_image(cv2.imread(filepath), transformer)

for i in range(10):
    st = time.time()
    features = model(imgs.to(device)).cpu().detach().numpy()
    print(time.time() - st)

# print(features)
print(features.shape)
