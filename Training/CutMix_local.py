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
########## Load Dataset
sys.path.append("..") ## to import parent's folder
from Data.BengaliDataset import BengaliDataset
from Local import DIR
########### YOUR DIR

def main():

    parser = argparse.ArgumentParser(
        description="Train Hifigan (See detail in examples/hifigan/train_hifigan.py)"
    )
    parser.add_argument(
    "--traindir",
    default=None,
    type=str,
    help="directory including training data. ",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
    "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
    "--pretrained",
    default="",
    type=str,
    nargs="?",
    help="path of .h5 melgan generator to load weights from",
    )
    args = parser.parse_args()


    ###Load Dataset => split
    if args.traindir is None:
        raise ValueError("Please specify --train-dir")
    else :
        X_train, X_val = train_test_split(pd.read_csv(f"{args.train_dir}/train.csv"), test_size = 0.2)
    
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
    if args.pretrained is None :
        raise ValueError("Please specify --train-dir")
    model = pretrainedmodels.__dict__[args.pretrained](pretrained='imagenet')
    in_features = model.last_linear.in_features
    model.last_linear = torch.nn.Linear(in_features, 186) 

    ###Prepare for training

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                      mode='max',
                                                      verbose=True,
                                                      patience=7,
                                                      factor=0.5)

    train_dataset = BengaliDataset(csv=X_train,
                            img_height=137,
                            img_width=236,
                            transform=train_augmentation)
    valid_dataset = BengaliDataset(csv=X_val,
                                img_height=137,
                                img_width=236,
                                transform=valid_augmentation)
    train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            num_workers=0,
                            batch_size=128
                        )
    valid_loader = DataLoader(valid_dataset,
                        shuffle=False,
                            num_workers=0,
                            batch_size=128
                        ) 

    train_model()
                                                     
if __name__ == "__main__":
    main()
