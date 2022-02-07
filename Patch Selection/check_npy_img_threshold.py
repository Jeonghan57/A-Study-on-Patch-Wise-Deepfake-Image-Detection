# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:01:38 2021

@author: user
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################# Adjustable Parameter #################

batch_size = 64

patch_size = 128

delta = 0.8 # upper threshold

########################################################

img_per_patch = int((256 / patch_size) ** 2)
data_size =  img_per_patch * 2000



def get_data(batch_size):
    path = "./Dataset(npy)/Celeb/"
    data_holder = np.zeros([data_size*2, 3, patch_size, patch_size], dtype=np.uint8)
    labels = np.zeros([data_size*2, 2])
    
    # temp = np.load(path + "test(StyleGAN2)_real.npy")
    temp = np.load(path + "test(PROGAN)("+ str(patch_size) +")_real.npy") # data_size, patch_size, patch_size, 3
    temp = np.reshape(temp, [data_size, 3, patch_size, patch_size])
    data_holder[:data_size, :, :, :] = temp[:data_size]
    
    # temp = np.load(path + "test(StyleGAN2)_fake.npy")
    temp = np.load(path + "test(PROGAN)("+ str(patch_size) +")_fake.npy") # data_size, patch_size, patch_size, 3
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


# Evaluate images with filtered patches in upper threshold only

def Evaluate_whole_image_ver1(Net):
    # save_path = "./result/Xception/"
    save_path = "./result/Xception(" + str(patch_size) + ")/"
    for i in range(50):
        Net.load_state_dict(torch.load(save_path + "epoch_" + str(i) + ".pth"), strict=False)
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
        threshold = (max(y_tt_a) - min(y_tt_a)) * (1 - delta)
        
        lib = []
        for l in range(len(y_tt_a)):
            
            if y_tt_a[l] > threshold:
                lib.append(l)
  
        # print(lib)
        # print("선택된 패치 개수 :", len(lib), "/", str(data_size * 2))
 
        correct_tt = 0
        incorrect_tt = 0
        acc = 0
        

        for j in range(0, data_size*2, img_per_patch):
            if j < data_size:
                for k in range(0, img_per_patch):
                    correct = 0
                    incorrect = 0
                    if j+k in lib:                
                        if (ys[j+k] == ypreds[j+k]):
                            correct += 1
                        else:
                            incorrect += 1
                    
                if correct > incorrect:
                    correct_tt += 1
                elif correct == 0 and incorrect == 0:
                    continue
                elif correct <= incorrect:
                    incorrect_tt +=1
            
            else:
                for k in range(0, img_per_patch):
                    correct = 0
                    incorrect = 0
                    if j+k in lib:
                        if (ys[j+k] == ypreds[j+k]):
                            correct += 1  
                        else:
                            incorrect += 1
                        
                if correct >= incorrect:
                    correct_tt += 1
                elif correct == 0 and incorrect == 0:
                    continue
                elif correct < incorrect:
                    incorrect_tt +=1
                
        acc = ((correct_tt) / (correct_tt + incorrect_tt)) * 100        
        
        print("epoch " + str(i))
        print("Correct:{0} / Incorrect:{1} / Accuracy:{2:.2f}%" .format(correct_tt, incorrect_tt, acc))

# Evaluate all images. If there aren't filtered patches in single image,
# then use all patches in single image

def Evaluate_whole_image_ver2(Net):
    # save_path = "./result/Xception/"
    save_path = "./result/Xception(" + str(patch_size) + ")/"
    for i in range(50):
        Net.load_state_dict(torch.load(save_path + "epoch_" + str(i) + ".pth"), strict=False)
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
        threshold = (max(y_tt_a) - min(y_tt_a)) * (1 - delta)
        
        lib = []
        for l in range(len(y_tt_a)):
            
            if y_tt_a[l] > threshold:
                lib.append(l)
  
        # print(lib)
        # print("선택된 패치 개수 :", len(lib), "/", str(data_size * 2))
 
        correct_tt = 0
        incorrect_tt = 0
        acc = 0
        

        for j in range(0, data_size*2, img_per_patch):
            if j < data_size:
                correct = 0
                incorrect = 0
                for k in range(0, img_per_patch):
                    if j+k in lib:
                        if (ys[j+k] == ypreds[j+k]):
                            correct += 1
                        else:
                            incorrect += 1
                    
                if correct > incorrect:
                    correct_tt += 1
                elif correct == 0 and incorrect == 0:
                    whole = (ys[j:j+img_per_patch] == ypreds[j:j+img_per_patch]).float().sum()
                    if whole > (img_per_patch/2):
                        correct_tt += 1
                    else:
                        incorrect_tt += 1
                elif correct <= incorrect:
                    incorrect_tt +=1
                    
            else:
                correct = 0
                incorrect = 0 
                for k in range(0, img_per_patch):
                    if j+k in lib:
                        if (ys[j+k] == ypreds[j+k]):
                            correct += 1
                        else:
                            incorrect += 1

                if correct >= incorrect:
                    correct_tt += 1
                elif correct == 0 and incorrect == 0:
                    whole = (ys[j:j+img_per_patch] == ypreds[j:j+img_per_patch]).float().sum() 
                    if whole >= (img_per_patch/2):
                        correct_tt += 1
                    else:
                        incorrect_tt += 1
                elif correct < incorrect:
                    incorrect_tt +=1
                
        acc = ((correct_tt) / (correct_tt + incorrect_tt)) * 100        
        
        print("epoch " + str(i))
        print("Correct:{0} / Incorrect:{1} / Accuracy:{2:.2f}%" .format(correct_tt, incorrect_tt, acc))
    


###################### Recall Fucntions ######################

# Evaluate_whole_image_ver1(net)
# Evaluate_whole_image_ver2(net)

##############################################################