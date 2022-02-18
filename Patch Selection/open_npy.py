# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:52:43 2021

@author: IVCML
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 텐서와 크기 확인
data = np.load('./Dataset(npy)/FFHQ/train(StyleGAN2)(128)_real.npy')

print(data)
print(data.shape)


cv2.imshow("o.png", data[0,:,:,:])

cv2.waitKey()
cv2.destroyAllWindows()

# 이미지 확인
"""
for i in range(10):
    plt.imshow(data[i,:,:,:])
    plt.show()
"""  
"""
for i in range(-1, -11, -1):
    plt.imshow(data[i,:,:,:])
    plt.show()
"""