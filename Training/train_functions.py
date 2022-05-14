from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import recall_score

########## Load Dataset
import sys
sys.path.append("..") ## to import parent's folder
from Local import DIR
from Data.BengaliDataset import BengaliDataset
########### YOUR DIR
def cut(W,H,lam):
        
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_accuracy(dataloader):
    gc.collect()
    torch.cuda.empty_cache()
    losses = []
    ground_true = []
    pred = []
    loss_fn = nn.CrossEntropyLoss()
    model.eval() # For later #

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for images, labels in dataloader :
        images, labels = images.to(device), labels.to(device)
        model = model.to(device)
        output = model(images)

        grapheme = output[:, :168]
        vowel = output[:, 168:179]
        cons = output[:, 179:]
        
        loss = loss_fn(grapheme, labels[:, 0]) + loss_fn(vowel, labels[:, 1]) + loss_fn(cons, labels[:, 2])
        losses.append(loss)

        grapheme = grapheme.cpu().argmax(dim=1).data.numpy()
        vowel = vowel.cpu().argmax(dim=1).data.numpy()
        cons = cons.cpu().argmax(dim=1).data.numpy()

        ground_true.append(labels.cpu().numpy())
        pred.append(np.stack([grapheme, vowel, cons], axis=1))
    
    ground_true = np.concatenate(ground_true)
    pred = np.concatenate(pred)

    loss = np.mean(losses)

    score_g = recall_score(ground_true[:, 0], pred[:, 0], average='macro')
    score_v = recall_score(ground_true[:, 1], pred[:, 1], average='macro')
    score_c = recall_score(ground_true[:, 2], pred[:, 2], average='macro')

    final_score = np.average([score_g, score_v, score_c], weights=[2, 1, 1])

    return [score_g, score_v, score_c, loss, final_score]

def train_model(model, train, valid, n_iters=500, learn_rate=0.001, batch_size=128, weight_decay=0, iscutmix=1):
  # Lists to store model's performance information
  iters, losses, val_acc, train_acc = [], [], [], []
  high_score = 0
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                      mode='max',
                                                      verbose=True,
                                                      patience=7,
                                                      factor=0.5)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"we are going to use {device}")
                                                  
    ##########
  for i in tqdm(range(n_iters)):
    for images, labels in tqdm(train):
      images, labels = images.to(device), labels.to(device)
      model = model.to(device)
      model.train()
      if (np.random.rand() < 0.5) and (iscutmix) :

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

        out = model(images)

        grapheme = out[:,:168] # output of grapheme
        vowel = out[:, 168:179] # output of grapheme
        cons = out[:, 179:] # output of grapheme
        
        loss1 = criterion(grapheme, targets_gra) * lam + criterion(grapheme, shuffled_targets_gra) * (1. - lam)
        loss2 = criterion(vowel, targets_vow) * lam + criterion(vowel, shuffled_targets_vow) * (1. - lam)
        loss3 = criterion(cons, targets_con) * lam + criterion(cons, shuffled_targets_con) * (1. - lam)
            
        loss = 0.5 * loss1 + 0.25 * loss2 + 0.25 * loss3
      else : # No CutMix
        out = model(images)        

        grapheme = out[:,:168] # output of grapheme
        vowel = out[:, 168:179] # output of grapheme
        cons = out[:, 179:] # output of grapheme
        loss = criterion(grapheme, labels[:, 0]) + criterion(vowel, labels[:, 1]) + criterion(cons, labels[:, 2])

      ##공통 부분##
      loss.backward()               # backward pass (compute parameter updates)
      optimizer.step()              # make the updates for each parameter
      optimizer.zero_grad()         # reset the gradients for the next iteration
      gc.collect()
      torch.cuda.empty_cache()
    # Save the current training and validation information at every 10th iteration
    if (i+1) % 2 == 0:
      iters.append(i)
      losses.append(float(loss)/batch_size)  
      ###########evaluate####################
      losses = []
      ground_true = []
      pred = []
      model.eval() # For later #

      for images, labels in valid :

          images, labels = images.to(device), labels.to(device)
          output = model(images)

          grapheme = output[:, :168]
          vowel = output[:, 168:179]
          cons = output[:, 179:]
          
          loss = criterion(grapheme, labels[:, 0]) + criterion(vowel, labels[:, 1]) + criterion(cons, labels[:, 2])
          losses.append(loss)

          grapheme = grapheme.cpu().argmax(dim=1).data.numpy()
          vowel = vowel.cpu().argmax(dim=1).data.numpy()
          cons = cons.cpu().argmax(dim=1).data.numpy()

          ground_true.append(labels.cpu().numpy())
          pred.append(np.stack([grapheme, vowel, cons], axis=1))
      
      ground_true = np.concatenate(ground_true)
      pred = np.concatenate(pred)

      loss = np.mean(losses)

      score_g = recall_score(ground_true[:, 0], pred[:, 0], average='macro')
      score_v = recall_score(ground_true[:, 1], pred[:, 1], average='macro')
      score_c = recall_score(ground_true[:, 2], pred[:, 2], average='macro')

      final_score = np.average([score_g, score_v, score_c], weights=[2, 1, 1])

      val_acc.append([score_g, score_v, score_c, loss, final_score])

      ##evaluate train acc#######
      for images, labels in train :

        images, labels = images.to(device), labels.to(device)
        output = model(images)

        grapheme = output[:, :168]
        vowel = output[:, 168:179]
        cons = output[:, 179:]
        
        loss = criterion(grapheme, labels[:, 0]) + criterion(vowel, labels[:, 1]) + criterion(cons, labels[:, 2])
        losses.append(loss)

        grapheme = grapheme.cpu().argmax(dim=1).data.numpy()
        vowel = vowel.cpu().argmax(dim=1).data.numpy()
        cons = cons.cpu().argmax(dim=1).data.numpy()

        ground_true.append(labels.cpu().numpy())
        pred.append(np.stack([grapheme, vowel, cons], axis=1))
      
      ground_true = np.concatenate(ground_true)
      pred = np.concatenate(pred)

      loss = np.mean(losses)

      score_g = recall_score(ground_true[:, 0], pred[:, 0], average='macro')
      score_v = recall_score(ground_true[:, 1], pred[:, 1], average='macro')
      score_c = recall_score(ground_true[:, 2], pred[:, 2], average='macro')

      final_score = np.average([score_g, score_v, score_c], weights=[2, 1, 1])

      train_acc.append([score_g, score_v, score_c, loss, final_score])

      #######################################
      if high_score < val_acc[-1][-1] :
        high_score = val_acc[-1][-1]
        PATH = './saved_models/best_model.pth'

        torch.save(model.state_dict(), PATH)
  
  write_log(val_acc, "val_history")
  write_log(train_acc, "train_history")
      

def write_log(logs, title) :

    file_name = f"./logs/{title}.txt"
    with open(file_name,'w') as f:
      f.write(f"\n-------{title}---------\n")
      for i, item in enumerate(logs) :
        f.write(f"{i}th\n")
        f.write(' '.join(item))
    f.close()
    print(f"file saved as {file_name}")

def plot(iters, losses, train_acc, val_acc) :

  print(f'Plotting')

  # Plotting Training Loss, Accuracy and Validation Accuracy
  plt.figure(figsize=(10,4))
  plt.subplot(1,2,1)
  plt.title("Training Curve")
  plt.plot(iters, losses, label="Train")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")

  plt.subplot(1,2,2)
  plt.title("Training Curve")
  plt.plot(iters, train_acc, label="Train")
  plt.plot(iters, val_acc, label="Validation")
  plt.xlabel("Iterations")
  plt.ylabel("Training Accuracy")
  plt.legend(loc='best')
  plt.show()
  print("Final Training Accuracy: {}".format(train_acc[-1]))
  print("Final Validation Accuracy: {}".format(val_acc[-1]))

if __name__ == "__main__" :

  import pandas as pd
  from sklearn.model_selection import train_test_split
  from torch.utils.data import DataLoader
  import torchvision.transforms as T
  from torchvision  import models

  df_train = pd.read_csv(f"{DIR}/train.csv")
  X_train, X_val = train_test_split(df_train, test_size=0.01)
  train_dataset = BengaliDataset(data=X_train,
                            img_height=137,
                            img_width=236,
                            transform=T.ToTensor())
  train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            num_workers=0,
                            batch_size=16
                            )
  valid_dataset = BengaliDataset(data=X_val,
                            img_height=137,
                            img_width=236,
                            transform=T.ToTensor())
  valid_loader = DataLoader(valid_dataset,
                        shuffle=False,
                          num_workers=0,
                          batch_size=16
                        )
  batch = next(iter(train_loader))
  images, labels = batch
  # VGG16 Model Loading
  use_pretrained = True
  model = models.resnet18(pretrained=use_pretrained)
  ## 우리 이미지 사이즈에 맞게 튜닝
  model.fc = torch.nn.Linear(model.fc.in_features, 186) 

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)
  # model = model.to(device)
  # for images, labels in train_loader :
  #   out = model.to(device)(images.to(device))
  #   break
  
