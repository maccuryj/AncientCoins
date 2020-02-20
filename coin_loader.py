import os
import numpy as np
import torch
import torchnet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
from sklearn.model_selection import train_test_split
from skimage import io


class AncientCoins():
    def __init__(self, root_dir, is_gpu, batch_size=32, workers=2, test_size=0.25):
        labels = {}
        X = []
        y = []                
        
        for i, d in enumerate(os.listdir(root_dir)):
            labels[i] = int(d)
            for f in os.listdir(os.path.join(root_dir, str(d))):       
                X.append(f)
                y.append(i)
        
        self.root_dir = root_dir     
        self.labels = labels

        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=test_size, stratify=y)
        self.trainset, self.valset = self.get_datasets()
        self.trainloader = self.get_dataloaders(self.trainset, batch_size, workers, is_gpu)
        self.valloader = self.get_dataloaders(self.valset, batch_size, workers, is_gpu)
        
    def check_exists():
        return self.root_dir in os.listdir()


    def get_labels(self):
        return self.labels

    def get_datasets(self):
        trainset = CoinsDataset(self.root_dir, self.train_X, self.train_y, self.labels)
        valset = CoinsDataset(self.root_dir, self.test_X, self.test_y, self.labels)
        
        return trainset, valset

    def get_dataloaders(self, dataset, batch_size, workers, is_gpu):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=is_gpu)
        return loader
        

class CoinsDataset(Dataset):  
    def __init__(self, root_dir, X, y, labels):
        self.X = X
        self.y = y
        self.root_dir = root_dir
        self.labels = labels
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        try:
            x = io.imread(os.path.join(self.root_dir, str(self.labels[self.y[idx]]), self.X[idx]))
        except (ValueError):
            print(os.path.join(self.root_dir, str(self.labels[self.y[idx]]), self.X[idx]))
            print(ValueError)
            
        X = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28,28)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])(x)
        y = self.y[idx]

        return X,y