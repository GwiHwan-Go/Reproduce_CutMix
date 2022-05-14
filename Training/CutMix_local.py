## You can run this code by python train.py [dir] ##
## to train and experiment with various variables
## input : preprocessed_dataset, 
##      params : pre_trained_model, whether to use CutMix.
## process
## output : trained model.pt file. This code will export the model.
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm.auto import tqdm
import os
import argparse
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.nn import Linear
from torchvision import models
########## Load Dataset
sys.path.append("..") ## to import parent's folder
from Data.BengaliDataset import BengaliDataset
from Local import DIR
from train_functions import get_accuracy, train_model
########### YOUR DIR

def main():
    ##############PARAMETERS##################
    batch_size = 128
    num_epochs = 2
    ##############PARAMETERS##################

    ###Set Augmentation
    train_augmentation = T.Compose([
        T.ToTensor(),
        T.RandomRotation(20),
        ##we can add more augmentation##
    ])

    valid_augmentation = T.Compose([
        T.ToTensor(),
    ])

    ###Choose pretrained model.

    model = models.resnet18(pretrained = True)
    ## 우리 이미지 사이즈에 맞게 튜닝
    model.fc = torch.nn.Linear(model.fc.in_features, 186) 

    ###Prepare for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                      mode='max',
                                                      verbose=True,
                                                      patience=7,
                                                      factor=0.5)
    df_train = pd.read_csv(f"{DIR}/train.csv")
    X_train, X_val = train_test_split(df_train, test_size=0.2)

    train_dataset = BengaliDataset(data=X_train,
                                img_height=137,
                                img_width=236,
                                transform=train_augmentation)
    train_loader = DataLoader(train_dataset,
                                shuffle=True,
                                num_workers=0,
                                batch_size=batch_size
                                )
    valid_dataset = BengaliDataset(data=X_val,
                                img_height=137,
                                img_width=236,
                                transform=valid_augmentation)
    valid_loader = DataLoader(valid_dataset,
                            shuffle=False,
                            num_workers=0,
                            batch_size=batch_size
                            )

    train_model(model, train_loader, valid_loader,
    n_iters=num_epochs, batch_size=batch_size)
                                                     
if __name__ == "__main__":
    main()
