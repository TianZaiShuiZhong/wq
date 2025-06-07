import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import config

class TennisBallDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.transform = transform or self.get_default_transform()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 随机生成条件向量(晴天/阴影/变形/半遮挡/污染)
        condition = torch.zeros(config.Config.num_conditions)
        # 随机选择1-2个条件
        selected = random.sample(range(config.Config.num_conditions), random.randint(1, 2))
        for i in selected:
            condition[i] = 1.0
            
        if self.transform:
            image = self.transform(image)
            
        return image, condition
    
    def get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((config.Config.image_size, config.Config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
