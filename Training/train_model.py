from tqdm.auto import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from CutMix import cut ## CutMix Algorithm

def get_accuracy(model, data):
    correct = 0
    total = 0
    model.eval() # For later #
    for images, labels in torch.utils.data.DataLoader(data, batch_size=64):
        output = model(images)

        grapheme = output[:, :168]
        vowel = output[:, 168:179]
        cons = output[:, 179:]

        grapheme = grapheme.argmax(dim=1).data.numpy()
        vowel = vowel.argmax(dim=1).data.numpy()
        cons = cons.argmax(dim=1).data.numpy()

        pred = output.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return 100.0 * correct / total

def train_model(model, train, valid, n_iters=500, learn_rate=0.001, batch_size=128, weight_decay=0, iscutmix=1):
  # Lists to store model's performance information
  iters, losses, train_acc, val_acc = [], [], [], []

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
  for i in trange(n_iters):
    for images, labels in iter(train):
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

      # Save the current training and validation information at every 10th iteration
      if (i+1) % 10 == 0:
          iters.append(i)
          losses.append(float(loss)/batch_size)        # compute *average* loss
          train_acc.append(get_accuracy(model, train)) # compute training accuracy 
          val_acc.append(get_accuracy(model, valid))   # compute validation accuracy


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
  highest=[0,0]
  for i, val_acc in zip(iters, val_acc):
    if val_acc>highest[1] :
      print(f"it achieved new high_acc at iter {i}th with {val_acc}%")
      highest = [i,val_acc]