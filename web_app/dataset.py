from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk

captions_file = "../data/train/captions_train.txt"
captions_df = pd.read_table(
    captions_file, delimiter=",", header=None, names=["image", "caption"]
)
captions_df.head()

# Get only unique images
unique_images = captions_df["image"].unique()

# Split images
train_images, test_images = train_test_split(
    unique_images, test_size=0.2, random_state=42
)
train_images, val_images = train_test_split(
    train_images, test_size=0.1, random_state=42
)

image_dir = '../data/test'

# DataFrame creation
train_df = captions_df[captions_df["image"].isin(train_images)].reset_index(drop=True)
train_df = train_df.dropna().reset_index(drop=True)
val_df = captions_df[captions_df["image"].isin(val_images)].reset_index(drop=True)
val_df = val_df.dropna().reset_index(drop=True)
test_df = captions_df[captions_df["image"].isin(test_images)].reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)

print(f"Numero di immagini nel set di addestramento: {len(train_images)}")
print(f"Numero di immagini nel set di validazione: {len(val_images)}")
print(f"Numero di immagini nel set di test: {len(test_images)}")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.word_freq = {}
        self.idx = 4
        # self.translator = str.maketrans("","", string.punctuation + string.digits + "\t\r\n")

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text)

    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                if word not in self.word_freq:
                    self.word_freq[word] = 1
                else:
                    self.word_freq[word] += 1

                if self.word_freq[word] == self.freq_threshold:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [
            self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokenized_text
        ]


vocab = Vocabulary(freq_threshold=5)
caption_list = train_df["caption"].tolist()
vocab.build_vocabulary(caption_list)

print(f"Dimensione del vocabolario: {len(vocab)}")


class FlickrDataset(Dataset):
    def __init__(self, dataframe, image_dir, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

        # Creiamo una lista di coppie (immagine, didascalia)
        self.image_ids = []
        self.captions = []

        grouped = dataframe.groupby("image")["caption"].apply(list).reset_index()

        for idx in range(len(grouped)):
            img_id = grouped.loc[idx, "image"]
            captions = grouped.loc[idx, "caption"]
            for cap in captions:
                self.image_ids.append(img_id)
                self.captions.append(cap)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.word2idx["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word2idx["<END>"])
        numericalized_caption = torch.tensor(numericalized_caption)

        return image, numericalized_caption


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Valori standard per ImageNet
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def retrieve_dataset():
    test_dataset = FlickrDataset(test_df, image_dir, vocab, transform=transform)
    return test_dataset
