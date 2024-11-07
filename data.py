import random
import torch
from torch.utils.data import DataLoader, Dataset

class ImageNetLabelDataset(Dataset):
    def __init__(self, categories):

        self.categories = categories
        # Pre-generate indices for reproducibility
        self.indices = list(range(self.categories))
        random.shuffle(self.indices)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
    
        x = torch.zeros(1000)
        y = torch.zeros(1000)
        
        i = self.indices[idx]
        x[i] = 1
        y[i] = 1
        
        return x, y

def get_dataloader(categories=10,batch_size=250,train=True):
    if train:
        train_ds = ImageNetLabelDataset(categories)
        train_loader = DataLoader(train_ds, batch_size=batch_size)
        return train_loader
    valid_ds = ImageNetLabelDataset(categories)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    return valid_loader
    


