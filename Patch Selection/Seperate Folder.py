# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:01:38 2021

@author: user
"""

import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################# Adjustable Parameter #################

batch_size = 256

patch_size = 32

theta = 0.9 # upper threshold

epoch = 49

########################################################

img_per_patch = int((256 / patch_size) ** 2)
data_size =  img_per_patch * 2000


def get_data(batch_size):
    path = "./Dataset(npy)/FFHQ/"
    data_holder = np.zeros([data_size*2, 3, patch_size, patch_size], dtype=np.uint8)
    labels = np.zeros([data_size*2, 2])
    
    # temp = np.load(path + "test(StyleGAN2)_real.npy")
    temp = np.load(path + "test(StyleGAN2)("+ str(patch_size) +")_real.npy") # data_size, patch_size, patch_size, 3
    temp = np.reshape(temp, [data_size, 3, patch_size, patch_size])
    data_holder[:data_size, :, :, :] = temp[:data_size]
    
    # temp = np.load(path + "test(StyleGAN2)_fake.npy")
    temp = np.load(path + "test(StyleGAN2)("+ str(patch_size) +")_fake.npy") # data_size, patch_size, patch_size, 3
    temp = np.reshape(temp, [data_size, 3, patch_size, patch_size])
    data_holder[data_size:, :, :, :] = temp[:data_size]
    
    labels[:data_size, 0] = 1
    labels[data_size:, 1] = 1
    
    data_holder = torch.from_numpy(data_holder).float()
    labels = torch.from_numpy(labels).long()
    ds = TensorDataset(data_holder, labels)
    del temp, labels
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    return data_loader


from XceptionNet import Xception
net = Xception(num_classes = 2)

def Seperate_img(Net):
    
    data_r = np.load('./Dataset(npy)/FFHQ/test(StyleGAN2)(' + str(patch_size) + ')_real.npy')
    data_f = np.load('./Dataset(npy)/FFHQ/test(StyleGAN2)(' + str(patch_size) + ')_fake.npy')
    
    save_path = "./result/Xception(" + str(patch_size) + ")/"

    Net.load_state_dict(torch.load(save_path + "epoch_" + str(epoch) + ".pth"), strict=False)
    Net = Net.to(device).eval()
        
    test_data = get_data(batch_size)
        
    # Test
    ys = []
    ypreds = []
    y_tt = []
    for X, Y in test_data:
        X = X.to(device)
        Y = Y.to(device)
        
        with torch.no_grad():
            
            y_predict = Net(X)
            _, y_pred = Net(X).max(1)
            for p in range(len(y_predict)):
                z = float(y_predict[p, 1]) - float(y_predict[p, 0])
                y_tt.append(z)
            
            ys.append(Y.max(1)[1])
            ypreds.append(y_pred)
               
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
        
    y_tt_a = np.abs(y_tt)
    threshold = (max(y_tt_a) - min(y_tt_a)) * (1 - theta)
    
    lib = []
    for l in range(len(y_tt_a)):
        
        if y_tt_a[l] > threshold:
            lib.append(l)

    # print(lib)
    # print("선택된 패치 개수 :", len(lib), "/", str(data_size * 2))

    for j in tqdm(range(0, data_size*2)):
        if j < data_size:
            if (ys[j] == ypreds[j]):
                if j in lib:
                    cv2.imwrite( "./Seperate img/threshold = " + str(theta) + "/StyleGAN2(" + str(patch_size) + ")/Well_predicted/real/" + str(j) + ".png", data_r[j, :, :, :])
                else:
                    cv2.imwrite( "./Seperate img/threshold = " + str(theta) + "/StyleGAN2(" + str(patch_size) + ")/Poor_predicted/real/" + str(j) + ".png", data_r[j, :, :, :])
            else:
                continue

        else:
            if (ys[j] == ypreds[j]):
                if j in lib:
                    cv2.imwrite( "./Seperate img/threshold = " + str(theta) + "/StyleGAN2(" + str(patch_size) + ")/Well_predicted/fake/" + str(j-data_size) + "(fake).png", data_f[j-data_size, :, :, :])
                else:
                    cv2.imwrite( "./Seperate img/threshold = " + str(theta) + "/StyleGAN2(" + str(patch_size) + ")/Poor_predicted/fake/" + str(j-data_size) + "(fake).png", data_f[j-data_size, :, :, :])
            else:
                continue
    

###################### Recall Fucntions ######################

Seperate_img(net)

##############################################################