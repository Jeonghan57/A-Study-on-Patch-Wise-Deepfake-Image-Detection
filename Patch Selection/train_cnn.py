# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 22:52:51 2021

@author: IVCML
"""

import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


patch_size = 256
img_per_patch = int((256 / patch_size) ** 2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #gpu 사용 가능?

def get_data(batch_size, train=True):
    data_size = 10000*4 if train else 2000*img_per_patch
    code = "train" if train else "test"
    path = "./Dataset(npy)/Celeb/"
    data_holder = np.zeros([data_size*2, 3, patch_size, patch_size], dtype=np.uint8)
    labels = np.zeros([data_size*2, 2])
    
    if train == True : 
        temp = np.load(path + code + "(StyleGAN2)(" + str(patch_size) + ")_real.npy") # 10000, 256, 256, 3
        temp = np.reshape(temp, [10000*img_per_patch, 3, patch_size, patch_size])
        data_holder[:data_size, :, :, :] = temp[:data_size]
        
    
        temp = np.load(path + code + "(StyleGAN2)(" + str(patch_size) + ")_fake.npy")
        temp = np.reshape(temp, [10000*img_per_patch, 3, patch_size, patch_size])
        data_holder[data_size:, :, :, :] = temp[:data_size]
        
        
    else : 
        temp = np.load(path + code + "(PROGAN)_real.npy")
        # temp = np.load(path + code + "(StyleGAN2)(" + str(patch_size) + ")_real.npy") # 2000, 256, 256, 3
        temp = np.reshape(temp, [2000*img_per_patch, 3, patch_size, patch_size])
        data_holder[:data_size, :, :, :] = temp[:data_size]
        
        temp = np.load(path + code + "(PROGAN)_fake.npy")
        # temp = np.load(path + code + "(StyleGAN2)(" + str(patch_size) + ")_fake.npy")
        temp = np.reshape(temp, [2000*img_per_patch, 3, patch_size, patch_size])
        data_holder[data_size:, :, :, :] = temp[:data_size]
    
    labels[:data_size, 0] = 1
    labels[data_size:, 1] = 1
    
    data_holder = torch.from_numpy(data_holder).float()
    labels = torch.from_numpy(labels).long()
    ds = TensorDataset(data_holder, labels)
    del temp, labels
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=train)
    return data_loader

def train(Net, batch_size):
    Net = Net.to(device)
    lr = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Net.parameters(), lr=lr)
    
    print("Training Start")
    
    train_data = get_data(batch_size, True)
    for epoch in range(50):

        for X, Y in tqdm.tqdm(train_data):
            X = X.to(device)
            Y = Y.to(device)
                    
            #예측 [P(a), P(b)]
            y_pred = Net(X)
            
            #loss 계산
            loss = loss_fn(y_pred, torch.max(Y, 1)[1])
            
            #학습
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch:{epoch}, loss={loss}")
        #파라미터 저장
        torch.save(Net.state_dict(), "./result/Xception(" + str(patch_size) + ")/epoch_" + str(epoch) + ".pth")
        
"""
from torchvision import models

net = models.efficientnet_b4(pretrained=False)

features = net.features

for params in net.features.parameters():
    params.requires_grad = False

net.classifier[1].out_features = 2
"""
"""
net = models.resnet50(pretrained=False)

for p in net.parameters():
    p.requires_grad = False

net.fc = nn.Linear(2048, 2)
"""

from XceptionNet import Xception
net = Xception(num_classes = 2)
 

def Evaluate_Networks(Net):
    save_path = "./result/Xception/"
    for i in range(50):
        Net.load_state_dict(torch.load(save_path + "epoch_" + str(i) + ".pth"), strict=False)
        Net = Net.to(device).eval()
        
        test_data = get_data(128, False)
        
        # Test
        ys = []
        ypreds = []
        for X, Y in test_data:
            X = X.to(device)
            Y = Y.to(device)
            
            with torch.no_grad():
                # Value, Indices >> Get Indices
                _, y_pred = Net(X).max(1)
                ys.append(Y.max(1)[1])
                ypreds.append(y_pred)
                
        ys = torch.cat(ys)
        ypreds = torch.cat(ypreds)
        acc = (ys == ypreds).float().sum() / len(ys)
        
        print("epoch " + str(i) + " Accuracy : ", acc.item())

# train(net, 128)
        
Evaluate_Networks(net)