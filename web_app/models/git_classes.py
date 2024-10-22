import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from pathlib import Path


class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, captions_file, processor):

        self.image_folder = image_folder
        self.processor = processor

        root_img, root_cpt = Path(image_folder), Path(captions_file)
        if not (root_img.exists() and root_img.is_dir()):
            raise ValueError(f"Data root '{root_img}' is invalid")
        
        if not (root_cpt.exists() and root_cpt.is_file()):
            raise ValueError(f"Data root '{root_cpt}' is invalid")

        self.df = pd.read_table(captions_file, sep=",", header=None, names=["image", "caption"], dtype='str')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_name = os.path.join(self.image_folder, self.df.iloc[idx, 0])
        image = Image.open(img_name)
        caption = self.df.iloc[idx, 1]
        
        encoding = self.processor(images=image, text=caption, padding="max_length", return_tensors="pt").to(device)
        encoding = {k: v.squeeze() for k,v in encoding.items()}

        return encoding, image