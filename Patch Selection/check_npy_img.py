# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:11:30 2021

@author: user
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patch_size = 128

img_per_patch = int((256 / patch_size) ** 2)
data_size =  img_per_patch * 2000

def get_data(batch_size):
    path = "./Dataset(npy)/Church(LSUN)/"
    data_holder = np.zeros([data_size*2, 3, patch_size, patch_size], dtype=np.uint8)
    labels = np.zeros([data_size*2, 2])
    
    temp = np.load(path + "test(StyleGAN2)_real.npy")
    # temp = np.load(path + "test(StyleGAN2)("+ str(patch_size) +")_real.npy") # data_size, patch_size, patch_size, 3
    temp = np.reshape(temp, [data_size, 3, patch_size, patch_size])
    data_holder[:data_size, :, :, :] = temp[:data_size]
    
    temp = np.load(path + "test(StyleGAN2)_fake.npy")
    # temp = np.load(path + "test(StyleGAN2)("+ str(patch_size) +")_fake.npy") # data_size, patch_size, patch_size, 3
    temp = np.reshape(temp, [data_size, 3, patch_size, patch_size])
    data_holder[data_size:, :, :, :] = temp[:data_size]
    # data_holder.shape -> (data_size*2, 3, patch_size, patch_size)
    
    labels[:data_size, 0] = 1
    labels[data_size:, 1] = 1
    # labels.shape -> (data_size*2, 2)

    # numpy 배열을 torch화 함.
    data_holder = torch.from_numpy(data_holder).float()
    labels = torch.from_numpy(labels).long()
       
    ds = TensorDataset(data_holder, labels)
    del temp, labels
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    return data_loader


from XceptionNet import Xception
net = Xception(num_classes = 2)


"""
# 이미지가 어느 정도 정답인지
def Evaluate_whole_image(Net):
    save_path = "./result/Xception(" + str(patch_size) + ")/"
    for i in range(50):
        Net.load_state_dict(torch.load(save_path + "epoch_" + str(i) + ".pth"), strict=False)
        Net = Net.to(device).eval()
        
        test_data = get_data(2048)
        
        # Test
        ys = []
        ypreds = []
        for X, Y in test_data:
            X = X.to(device)
            Y = Y.to(device)
            
            with torch.no_grad():
                _, y_pred = Net(X).max(1)
                ys.append(Y.max(1)[1])
                ypreds.append(y_pred)
               
        ys = torch.cat(ys)
        ypreds = torch.cat(ypreds)
        
        
        correct = 0
        unknown = 0
        incorrect = 0
        acc = 0
        
        for j in range(0, data_size*2, img_per_patch):
            whole = (ys[j:j+img_per_patch] == ypreds[j:j+img_per_patch]).float().sum()
                
                
            if whole > (img_per_patch/2):
                correct += 1
                
            elif whole == (img_per_patch/2):
                unknown += 1
                
            elif whole < (img_per_patch/2):
                incorrect += 1
                
                
            del whole
        
        acc = ((correct) / (correct + unknown + incorrect)) * 100
        
        print("epoch " + str(i))
        print("Correct:{0} / Unknown:{1} / Incorrect:{2} / Accuracy:{3:.2f}%" .format(correct, unknown, incorrect, acc))
"""  
"""
# Real과 Fake의 수가 같으면 Fake인 이미지라고 판단하게 하여 정확도 분석

def Evaluate_whole_image(Net):
    # save_path = "./result/Xception/"
    save_path = "./result/Xception(" + str(patch_size) + ")/"
    for i in range(50):
        Net.load_state_dict(torch.load(save_path + "epoch_" + str(i) + ".pth"), strict=False)
        Net = Net.to(device).eval()
        
        test_data = get_data(512)
        
        # Test
        ys = []
        ypreds = []
        for X, Y in test_data:
            X = X.to(device)
            Y = Y.to(device)
            
            with torch.no_grad():
                _, y_pred = Net(X).max(1)
                ys.append(Y.max(1)[1])
                ypreds.append(y_pred)
               
        ys = torch.cat(ys)
        ypreds = torch.cat(ypreds)
        
        correct = 0
        incorrect = 0
        acc = 0
        
        for j in range(0, data_size*2, img_per_patch):
            whole = (ys[j:j+img_per_patch] == ypreds[j:j+img_per_patch]).float().sum()
            if j < data_size:                
                if whole > (img_per_patch/2):
                    correct += 1
                else:
                    incorrect += 1
            else:
                if whole >= (img_per_patch/2):
                    correct += 1
                else:
                    incorrect += 1

        acc = ((correct) / (correct + incorrect)) * 100
        
        print("epoch " + str(i))
        print("Correct:{0} / Incorrect:{1} / Accuracy:{2:.2f}%" .format(correct, incorrect, acc))
        
Evaluate_whole_image(net)
        
def Evaluate_Random_image(Net):
    save_path = "./result/Xception(" + str(patch_size) + ")/"
    for i in range(50):
        Net.load_state_dict(torch.load(save_path + "epoch_" + str(i) + ".pth"), strict=False)
        Net = Net.to(device).eval()
        
        test_data = get_data(512)
        
        # Test
        ys = []
        ypreds = []
        for X, Y in test_data:
            X = X.to(device)
            Y = Y.to(device)
            
            with torch.no_grad():
                _, y_pred = Net(X).max(1)
                ys.append(Y.max(1)[1])
                ypreds.append(y_pred)
               
        ys = torch.cat(ys)
        ypreds = torch.cat(ypreds)

        correct = 0
        incorrect = 0
        acc = 0
        
        for j in range(0, data_size*2, img_per_patch):
            whole = (ys[j:j+img_per_patch] == ypreds[j:j+img_per_patch]).float().sum()

            if whole > (img_per_patch/2):
                correct += 1

            else:
                incorrect += 1

        
        acc = ((correct) / (correct + incorrect)) * 100
        
        print("epoch " + str(i))
        print("Correct:{0} / Incorrect:{1} / Accuracy:{2:.2f}%" .format(correct, incorrect, acc))

# Evaluate_Random_image(net)
"""