# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:52:43 2021

@author: IVCML
"""

import numpy as np
from matplotlib import pyplot as plt

# 텐서와 크기 확인
data = np.load('./Dataset(npy)/Patch(Random_mix)/test(StyleGAN2)_real.npy')

print(data)
print(data.shape)

# 이미지 확인
for i in range(10):
    plt.imshow(data[i,:,:,:])
    plt.show()
    
"""
for i in range(-1, -11, -1):
    plt.imshow(data[i,:,:,:])
    plt.show()
"""