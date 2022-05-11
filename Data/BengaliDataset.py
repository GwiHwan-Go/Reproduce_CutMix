from torch.utils.data import Dataset
import joblib
import sys
import numpy as np
import sys
sys.path.append("..") ## to import parent's folder
from Local import DIR 
########### YOUR DIR

class BengaliDataset(Dataset):
    def __init__(self, csv, img_height, img_width, transform=None):
        self.csv = csv.reset_index()
        self.img_ids = csv['image_id'].values
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = joblib.load(f'{DIR}/train_images/{img_id}.pkl')
        img = img.reshape(self.img_height, self.img_width).astype(np.uint8)
        #img = 255 - img
        img = 255 - img
        #img = np.expand_dims(img, axis=2)
        img = img[:, :, np.newaxis]

        
        img = np.repeat(img, 3, 2)
        if self.transform is not None:
            img = self.transform(img)
        
        label_1 = self.csv.iloc[index].grapheme_root #-167
        label_2 = self.csv.iloc[index].vowel_diacritic #-10
        label_3 = self.csv.iloc[index].consonant_diacritic #-6
        
        return img, np.array([label_1, label_2, label_3])

if __name__ == "__main__" :
    import pandas as pd
    from torch.utils.data import DataLoader

    df = pd.read_csv(f"{DIR}/train.csv")
    dataset = BengaliDataset(csv=df,
                            img_height=137,
                            img_width=236
                            )
    print(dataset.index)