import os
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, datafolder, transform):
        self.datafolder = datafolder
        self.transform = transform
        self.image_list = [s for s in os.listdir(datafolder)]

    def __len__(self):
        return len(os.listdir(self.datafolder))

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.datafolder, self.image_list[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_name
