import os
import torch
import cv2
import numpy as np
from torchvision import transforms, utils, models
import torch.nn as nn
from tqdm import tqdm
from utils.data_process import preprocess_img, postprocess_img
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

flag = 1 # 0 for TranSalNet_Dense, 1 for TranSalNet_Res

import torch
print(torch.cuda.is_available())

if flag:
    from TranSalNet_Res import TranSalNet
    model = TranSalNet()
    model.load_state_dict(torch.load(r'pretrained_models\TranSalNet_Res.pth'))
else:
    from TranSalNet_Dense import TranSalNet
    model = TranSalNet()
    model.load_state_dict(torch.load(r'pretrained_models\TranSalNet_Dense.pth'))

model = model.to(device) 
model.eval()

test_img = r'example/sample_1.png' 

img = preprocess_img(test_img) # padding and resizing input image into 384x288
img = np.array(img)/255.
img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
img = torch.from_numpy(img)
img = img.type(torch.cuda.FloatTensor).to(device)
pred_saliency = model(img)
toPIL = transforms.ToPILImage()
pic = toPIL(pred_saliency.squeeze())

pred_saliency = postprocess_img(pic, test_img) # restore the image to its original size as the result

cv2.imwrite(r'example/result.png', pred_saliency, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) # save the result
print('Finished, check the result at: {}'.format(r'example/result.png'))

