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

        root_img, root_cpt = Path(image_folder), Path(captions_file)
        if not (root_img.exists() and root_img.is_dir()):
            raise ValueError(f"Data root '{root_img}' is invalid")
        
        if not (root_cpt.exists() and root_cpt.is_file()):
            raise ValueError(f"Data root '{root_cpt}' is invalid")

        self.image_folder = image_folder
        self.transform = transform
        self.df = pd.read_csv(captions_file, sep=",", header=None, names=["image", "caption"])
        self.img_groups = self.df.groupby("image")["caption"].apply(list).reset_index()
        
    def __len__(self):
        return len(self.img_groups)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.img_groups.iloc[idx, 0])
        print(img_name)
        image = Image.open(img_name)
        captions = self.img_groups.iloc[idx, 1]
        
        if self.transform: image = self.transform(image)
        
        return image, captions


def display_image(axis, image_tensor):

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(
            "The `display_image` function expects a `torch.Tensor` "
            + "use the `ToTensor` transformation to convert the images to tensors."
        )

    image_data = image_tensor.permute(1, 2, 0).numpy()
    height, width, _ = image_data.shape
    axis.imshow(image_data)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)


def compare_transforms(transformations, index):

    if not all(isinstance(transf, Dataset) for transf in transformations):
        raise TypeError(
            "All elements in the `transformations` list need to be of type Dataset"
        )

    num_transformations = len(transformations)
    fig, axes = plt.subplots(1, num_transformations)

    if num_transformations == 1:
        axes = [axes]

    for counter, (axis, transf) in enumerate(zip(axes, transformations)):
        axis.set_title(f"transf: {counter}")
        image_tensor = transf[index][0]
        display_image(axis, image_tensor)

    plt.show()

