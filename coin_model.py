import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



class BasicCNN(nn.Module):
    def __init__(self, num_classes, img_size):
        super(BasicCNN, self).__init__()
        out1 = 32
        out2 = 64
        conv_kernel_size = 5
        p = 2
        remaining_res = int((int((img_size-conv_kernel_size+1)/p)-conv_kernel_size+1)/p)

        self.spatial_resolution = out2*remaining_res**2

        self.model = nn.Sequential(
            nn.Conv2d(1, out1, conv_kernel_size),
            nn.ReLU(True),       
            nn.MaxPool2d(p, p),

            nn.Conv2d(out1, out2, conv_kernel_size),
            nn.ReLU(True),
            nn.MaxPool2d(p, p))

        self.fc = nn.Linear(self.spatial_resolution, num_classes)   

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.spatial_resolution)

        return self.fc(x)

    def feature_map(self, x):
        x = self.model(x)
        #Return flattened view?

        return x

class ResNet():
    def __init__(self, num_classes, saved_instance=False, filename = None, l=18, freeze=7):
        self.num_classes = num_classes
        if l not in (18,50):
            raise ValueError("No Resnet Model available with {0} layers".format(l))
        if l == 18:
            self.model = models.resnet18(pretrained=True)
            self._fine_tune(l, freeze)
        if l == 50:
            self.model = models.resnet50(pretrained=True)
            self._fine_tune(l, freeze)

    def _fine_tune(self, l, freeze):
        #Store dimension of feature output before last FC layer
        self.num_features = self.model.fc.in_features
        #Replace last FC layer to fit our problem
        self.model.fc = nn.Linear(self.num_features, self.num_classes)
        #Freeze set of layers
        for i, child in enumerate(self.model.children()):
            if i < freeze:
                for p in child.parameters():
                    p.requires_grad = False

    def feature_extraction(self, dataloader):
            modules = list(self.model.children())[:-1]
            self.extraction_model = nn.Sequential(*modules)
            for p in self.extraction_model.parameters():
                p.requires_grad = False

            self.extraction_model.eval()
            idxs = []
            output = []
            targets = []
            for i, (idx, inp, target) in enumerate(dataloader):
                idxs.append(idx)
                output.append(self.extraction_model(inp))
                targets.append(target)
                
            idxs = np.array(torch.cat(idxs))
            output = np.array(torch.cat(output)).reshape(-1, self.num_features)
            targets = np.array(torch.cat(targets))

            return idxs, output, targets
        


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class TrainModel():
    def __init__(self, device, train_loader, val_loader, model, criterion=None, optimizer=None, epochs=30):
        """
        Parameters:
            train_loader (torch.utils.data.DataLoader): The trainset dataloader
            val_loader (torch.utils.data.DataLoader): The validation or testset dataloader
            model (torch.nn.module): Model to be trained
            criterion (torch.nn.criterion): Loss function
            optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
            device (string): cuda or cpu

        """

        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=500).to(device)
        else: self.criterion=criterion
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                            betas=(0.5, 0.999))
        else: self.optimizer=optimizer

        for epoch in range(epochs):
            print("EPOCH:", epoch + 1)
            print("TRAIN")
            self.train()
            print("VALIDATION")
            self.validate()

    def train(self):
        """
        Trains/updates the model for one epoch on the training dataset.        
        """

        # create instances of the average meter to track losses and accuracies
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.train()

        for i, (idx, inp, target) in enumerate(self.train_loader):
            inp = inp.to(self.device)
            target = target.to(self.device)

            output = self.model(inp)

            loss = self.criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), inp.size(0))
            top1.update(prec1.item(), inp.size(0))

            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()

            # print the loss every 100 mini-batches
            if i % 100 == 0:
                print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    loss=losses, top1=top1))

    def validate(self):
        """
        Evaluates/validates the model
        """

        # create instances of the average meter to track losses and accuracies
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for i, (idx, inp, target) in enumerate(self.val_loader):
                inp = inp.to(self.device)
                target = target.to(self.device)
                
                output = self.model(inp)

                loss = self.criterion(output, target)

                prec1, _ = accuracy(output, target, topk=(1, 2))
                losses.update(loss.item(), inp.size(0))
                top1.update(prec1.item(), inp.size(0))

    
        print(' * Validation accuracy: Prec@1 {top1.avg:.3f} '.format(top1=top1))


