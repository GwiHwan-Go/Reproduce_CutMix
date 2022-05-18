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
import gc
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics import recall_score
########## Load Dataset
sys.path.append("..") ## to import parent's folder
from Data.BengaliDataset import BengaliDataset
from Local import DIR
from train_functions import cut, write_log
########### YOUR DIR

def main():
    ##############PARAMETERS##################
    batch_size = 128
    num_epochs = 40
    file_name = f"./logs/ex3-2.txt"
    iscutmix = 0
    ##############PARAMETERS##################

    ###Set Augmentation
    train_augmentation = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ##we can add more augmentation##
    ])

    valid_augmentation = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    ###Choose pretrained model.

    model = models.resnet18(pretrained = True)
    ## 우리 이미지 사이즈에 맞게 튜닝
    model.fc = torch.nn.Linear(model.fc.in_features, 186) 

    ###Prepare for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"we are going to use {device}")
    print(f"log file will be {file_name}")
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
#########################################################train##########################################

    best_score = 0

    for e in range(num_epochs):
        train_loss = []
        model.train()

        print(f"{e+1}th epoch traing started")
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            if (np.random.rand() < 0.5) and (iscutmix) : # Implement cutMix 

                logits = model(images)
                grapheme = logits[:, :168]
                vowel = logits[:, 168:179]
                cons = logits[:, 179:]

                loss = loss_fn(grapheme, labels[:, 0]) + loss_fn(vowel, labels[:, 1]) + loss_fn(cons, labels[:, 2])

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                train_loss.append(loss.item())

            else:
                
                lam = np.random.beta(1.0, 1.0) 
                rand_index = torch.randperm(images.size()[0])
                
                targets_gra = labels[:, 0]
                targets_vow = labels[:, 1]
                targets_con = labels[:, 2]
                
                shuffled_targets_gra = targets_gra[rand_index]
                shuffled_targets_vow = targets_vow[rand_index]
                shuffled_targets_con = targets_con[rand_index]
                
                bbx1, bby1, bbx2, bby2 = cut(images.size()[2], images.size()[3], lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                
                logits = model(images) #forward pass

                grapheme = logits[:,:168] # output of grapheme
                vowel = logits[:, 168:179] # output of grapheme
                cons = logits[:, 179:] # output of grapheme
                
                loss1 = loss_fn(grapheme, targets_gra) * lam + loss_fn(grapheme, shuffled_targets_gra) * (1. - lam)
                loss2 = loss_fn(vowel, targets_vow) * lam + loss_fn(vowel, shuffled_targets_vow) * (1. - lam)
                loss3 = loss_fn(cons, targets_con) * lam + loss_fn(cons, shuffled_targets_con) * (1. - lam)
                
                loss = 0.5 * loss1 + 0.25 * loss2 + 0.25 * loss3
                
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                train_loss.append(loss.item())
                
                
        val_loss = []
        val_true = []
        val_pred = []

        model.eval()
        print("starting validating the result")
        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader):
                inputs = inputs.cuda()
                targets = targets.cuda()

                logits = model(inputs)

                grapheme = logits[:, :168]
                vowel = logits[:, 168:179]
                cons = logits[:, 179:]

                loss = loss_fn(grapheme, targets[:, 0]) + loss_fn(vowel, targets[:, 1]) + loss_fn(cons, targets[:, 2])

                val_loss.append(loss.item())

                grapheme = grapheme.cpu().argmax(dim=1).data.numpy()
                vowel = vowel.cpu().argmax(dim=1).data.numpy()
                cons = cons.cpu().argmax(dim=1).data.numpy()

                val_true.append(targets.cpu().numpy())
                val_pred.append(np.stack([grapheme, vowel, cons], axis=1))

        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)

        val_loss = np.mean(val_loss)
        train_loss = np.mean(train_loss)

        score_g = recall_score(val_true[:, 0], val_pred[:, 0], average='macro')
        score_v = recall_score(val_true[:, 1], val_pred[:, 1], average='macro')
        score_c = recall_score(val_true[:, 2], val_pred[:, 2], average='macro')

        final_score = np.average([score_g, score_v, score_c], weights=[2, 1, 1])

        print(f'train_loss: {train_loss:.5f}; val_loss: {val_loss:.5f}; score: {final_score:.5f}')
        print(f'score_g: {score_g:.5f}; score_v: {score_v: .5f}, score_c: {score_c: .5f}')

        with open(file_name, 'a') as f:
            f.write(f'train_loss: {train_loss:.5f}; val_loss: {val_loss:.5f}; score: {final_score:.5f}\n')
            f.write(f'score_g: {score_g:.5f}; score_v: {score_v: .5f}, score_c: {score_c: .5f}\n')
        f.close()
        print(f"file saved as {file_name}")

        if final_score > best_score:
            best_score = final_score

            state_dict = model.cpu().state_dict()
            model = model.cuda()
            torch.save(state_dict, "./saved_models/model.pt")  
                       
if __name__ == "__main__":
    main()
