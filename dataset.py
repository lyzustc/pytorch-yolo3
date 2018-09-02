import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from image import *
from PIL import Image
from torchvision import transforms
import cv2

class MyDataset(Dataset):
    def __init__(self, rootpath, txtfile, dataset = 'voc', shape = (416,416), is_train = False, use_da = False, transform = transforms.ToTensor(), target_transform = None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        self.shape = shape
        with open(os.path.join(rootpath, txtfile)) as file:
            self.lines = file.readlines()
        self.len = len(self.lines)
        self.use_da = use_da
        self.rootpath = rootpath
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        assert index <= self.len, 'index out of range'
        imgpath = os.path.join(self.rootpath, self.lines[index].rstrip())
        assert os.path.exists(imgpath), 'image does not exist'
        
        if self.dataset == "voc":
            labpath = imgpath.replace("JPEGImages","labels").replace(".jpg",".txt")
        elif self.dataset == "coco":
            labpath = imgpath.replace("images","labels").replace(".jpg",".txt")
        else:
            #labpath = self.get_labpath(imgpath)
            labpath = imgpath.replace("Images","labels").replace(".jpg",".txt") 

        if self.is_train and self.use_da:
            img, label = load_data_detection(imgpath, labpath, self.shape, jitter = 0.2, hue = 0.1, saturation = 1.5, exposure = 1.5)
            label = torch.from_numpy(label)
            
        else:
            img = cv2.imread(imgpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.shape is not None:
                img = cv2.resize(img, self.shape)
            assert os.path.exists(labpath), 'label does not exist'
            label = torch.zeros(50*5)
            truths = np.loadtxt(labpath)
            truths = torch.from_numpy(truths).view(-1)
            label[0:truths.numel()] = truths
            
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)
    
    def get_labpath(self,imgpath):
        return None