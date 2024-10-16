import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import torch
import os

# Custom dataset class for image captioning
class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, captions_file, transform=None):

        self.image_folder = image_folder
        self.transform = transform

        root_img, root_cpt = Path(image_folder), Path(captions_file)
        if not (root_img.exists() and root_img.is_dir()):
            raise ValueError(f"Data root '{root_img}' is invalid")
        
        if not (root_cpt.exists() and root_cpt.is_file()):
            raise ValueError(f"Data root '{root_cpt}' is invalid")

        self.df = pd.read_table(captions_file, sep=",", header=None, names=["image", "caption"], dtype='str')
        self.img_groups = self.df.groupby("image")["caption"].apply(list).reset_index()
        
    def __len__(self):
        return len(self.img_groups)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.img_groups.iloc[idx, 0])
        image = Image.open(img_name)
        captions = self.img_groups.iloc[idx, 1]
        
        if self.transform: image = self.transform(image)
        
        return image, captions, img_name

