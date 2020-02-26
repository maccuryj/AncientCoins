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
    """
    Dataset creation class, providing on instantiation a stratified train/test split.
    Provides coin specific datasets and dataloaders for easy integration with PyTorch.

    Attributes:
        root_dir (str):             Root directory holding the image data
        labels (Dict):              Mapping dictionary for labels (if original labels are not (0...N))
        trainset (CoinDataset):     Training coin set
        valset (CoinDataset):       Validation coin set
        trainloader (DataLoader):   Training coin data loader
        valloader (DataLoader):     Validation coin data loader
    """
    def __init__(self, root_dir, img_size, is_gpu, batch_size=32, shuffle=True, workers=2, test_size=0.25, transform=None):
        """
        Instantation of datasets and dataloaders

        Args:
            img_size (int):                 Final width and height of images
            is_gpu (str):                   Description of CUDA device
            batch_size (int):               Size of mini-batches delivered by data loader
            shuffle (boolean):              Shuffle indices of data points
            workers (int):                  Parallelisation factor
            test_size (float):              Fraction of data used for validation
            transform (list(transforms)):   List of image transformations
        """
        labels = {}
        X = []
        y = []                
        
        #Pull the image classes and filenames from the data folder 
        for i, d in enumerate(os.listdir(root_dir)):
            labels[i] = int(d)
            for f in os.listdir(os.path.join(root_dir, str(d))):       
                X.append(f)
                y.append(i)
        
        self.root_dir = root_dir     
        self.labels = labels

        # Stratified train test split
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=test_size, stratify=y)

        #Set up default transform if none was passed
        if transform is None:
            transform = [
            transforms.ToPILImage(),
            transforms.Resize((img_size,img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()]

        # Create training and validation set as defined in CoinsDataset
        self.trainset = CoinsDataset(self.root_dir, self.train_X, self.train_y, self.labels, img_size, transform)
        self.valset = CoinsDataset(self.root_dir, self.test_X, self.test_y, self.labels, img_size, transform)        
        # Create PyTorch Dataloaders for coin data
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=is_gpu)
        self.valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=is_gpu)
        
    def check_exists(self):
        """
        Return whether the data directory exists
        """
        return self.root_dir in os.listdir()
        

class CoinsDataset(Dataset):
    """
    Coin Dataset class which handles the provision of tensors
    on the basis of images in the data folder structure

    Attributes:
        X (list(str)):          Filenames of coin images
        y (list(int)):          Class labels of coin images
    """
    def __init__(self, root_dir, X, y, labels, img_size, transform):
        self.X = X
        self.y = y
        self.root_dir = root_dir
        self.labels = labels
        self.img_size = img_size
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        try:
            x = io.imread(os.path.join(self.root_dir, str(self.labels[self.y[idx]]), self.X[idx]))
        except (ValueError):
            print(os.path.join(self.root_dir, str(self.labels[self.y[idx]]), self.X[idx]))
            print(ValueError)

        X = transforms.Compose(self.transform)(x)
        y = self.y[idx]

        return X,y
