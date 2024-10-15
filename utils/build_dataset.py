from torchvision.transforms import RandomHorizontalFlip, Resize, RandomResizedCrop
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
from utils.dataset import ImageCaptionDataset
from torch.utils.data import DataLoader

IMG_SIZE = 224
BATCH_SIZE = 64

transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_datasets(BATCH_SIZE=BATCH_SIZE, IMG_SIZE=IMG_SIZE, transform=transform):

    train_dataset = ImageCaptionDataset("../data/train", "../data/train/captions_train.txt", transform)
    val_dataset = ImageCaptionDataset("../data/val", "../data/val/captions_val.txt", transform)
    test_dataset = ImageCaptionDataset("../data/test", "../data/test/captions_test.txt", transform)

    small_train_dataset = ImageCaptionDataset("../data/train_small", "../data/train_small/small_captions_train.txt", transform)
    small_val_dataset = ImageCaptionDataset("../data/val_small", "../data/val_small/small_captions_val.txt", transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    small_train_dataloader = DataLoader(small_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    small_val_dataloader = DataLoader(small_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, small_train_dataloader, small_val_dataloader