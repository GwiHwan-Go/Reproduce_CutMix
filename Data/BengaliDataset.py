from torch.utils.data import Dataset
import joblib
import sys
import numpy as np
import sys
sys.path.append("..") ## to import parent's folder
from Local import DIR 
########### YOUR DIR

class BengaliDataset(Dataset):
    def __init__(self, data, img_height, img_width, transform=None):
        self.data = data.reset_index()
        self.img_ids = data['image_id'].values
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = joblib.load(f'{DIR}/train_images/{img_id}.pkl')
        img = img.reshape(self.img_height, self.img_width).astype(np.uint8)
        img = 255 - img
        img = img[:, :, np.newaxis]

        
        img = np.repeat(img, 3, 2)
        if self.transform is not None:
            img = self.transform(img)
        
        label_1 = self.data.iloc[index].grapheme_root #-167
        label_2 = self.data.iloc[index].vowel_diacritic #-10
        label_3 = self.data.iloc[index].consonant_diacritic #-6
        
        return img, np.array([label_1, label_2, label_3])

if __name__ == "__main__" :
    import pandas as pd
    from torch.utils.data import DataLoader

    df = pd.read_csv(f"{DIR}/train.csv")
    dataset = BengaliDataset(data=df,
                            img_height=137,
                            img_width=236
                            )
    print(len(dataset))