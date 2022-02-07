# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 03:25:22 2021

@author: user
"""

import cv2
import numpy as np
import glob
import os
from tqdm import tqdm


patch_size = 16

#################################################################################################################

patch_number = int((256 / patch_size) ** 2)

label = "real" 

# images_path = "./Datafolder/FFHQ/Partial Synthesis/test(StyleGAN2)/" + label
images_path = "./Datafolder/FFHQ/Partial Synthesis/test(StyleGAN2)(" + str(patch_size) + ")/" + label
image_lists = glob.glob(images_path + "/*.png")

data_holder = np.zeros([2000 * patch_number, patch_size, patch_size, 3], dtype=np.uint8)
i=0
for img_name in tqdm(os.listdir(images_path)):
    img = cv2.imread(os.path.join(images_path, img_name))
    data_holder[i, :, :, :] = img
    i += 1

# np.save("./Dataset(npy)/Patch(Random_mix)/test(StyleGAN2)_" + label + ".npy", data_holder)
np.save("./Dataset(npy)/Patch(Random_mix)/test(StyleGAN2)("+ str(patch_size) +")_" + label + ".npy", data_holder)

#################################################################################################################

label = "fake"

# images_path = "./Datafolder/FFHQ/Partial Synthesis/test(StyleGAN2)/" + label
images_path = "./Datafolder/FFHQ/Partial Synthesis/test(StyleGAN2)(" + str(patch_size) + ")/" + label
image_lists = glob.glob(images_path + "/*.png")

data_holder = np.zeros([2000 * patch_number, patch_size, patch_size, 3], dtype=np.uint8)
i = 0
for img_name in tqdm(os.listdir(images_path)):
    img = cv2.imread(os.path.join(images_path, img_name))
    data_holder[i, :, :, :] = img
    i += 1

# np.save("./Dataset(npy)/Patch(Random_mix)/test(StyleGAN2)_" + label + ".npy", data_holder)
np.save("./Dataset(npy)/Patch(Random_mix)/test(StyleGAN2)("+ str(patch_size) +")_" + label + ".npy", data_holder)
