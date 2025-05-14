import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class EdgeCornerDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.canny_dir = os.path.join(root_dir, 'canny')
        self.harris_dir = os.path.join(root_dir, 'harris')
        self.filenames = sorted(os.listdir(self.rgb_dir))
        self.transform = transform

        # .ToTensor() allows converting from [H, W, C] to [C, H, W] and scaling to [0,1]
        self.default_transform = T.Compose([
            T.ToTensor(),  
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        rgb_path = os.path.join(self.rgb_dir, file_name)
        canny_path = os.path.join(self.canny_dir, file_name)
        harris_path = os.path.join(self.harris_dir, file_name)

        image = Image.open(rgb_path).convert('RGB')
        canny = Image.open(canny_path).convert('L')  
        harris = Image.open(harris_path).convert('L')

        image = self.default_transform(image)
        canny = self.default_transform(canny)
        harris = self.default_transform(harris)

        # Output format:
        #   labels: [2, H, W] → channel 0: edge, channel 1: corner
        labels = torch.cat([canny, harris], dim = 0)

        return image, labels
