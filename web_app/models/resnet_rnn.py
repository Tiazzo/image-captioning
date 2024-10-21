import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import torchvision.models as models
import string
import nltk

nltk.download("punkt_tab")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset
captions_file = "data/train/captions_train.txt"
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


# Vocabulary class
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


# Function to load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    print(f"Caricati {len(embeddings)} vettori di embedding da GloVe.")
    return embeddings


# Caricamento degli embeddings GloVe
glove_file = "glove/glove.6B.100d.txt"  # Sostituisci con il percorso corretto
glove_embeddings = load_glove_embeddings(glove_file)


def create_embedding_matrix(vocab, glove_embeddings, embedding_dim):
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in vocab.word2idx.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
        else:
            # Inizializza con un vettore casuale per le parole non trovate
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.tensor(embedding_matrix, dtype=torch.float32)


embedding_dim = (
    100  # Deve corrispondere alla dimensione degli embeddings GloVe scaricati
)
embedding_matrix = create_embedding_matrix(vocab, glove_embeddings, embedding_dim)


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

transf_totensor = transforms.Compose([transforms.ToTensor()])

image_dir = "data/train"

train_dataset = FlickrDataset(train_df, image_dir, vocab, transform=transform)
test_dataset = FlickrDataset(test_df, image_dir, vocab, transform=transform)


def collate_fn(batch):
    images = []
    captions = []
    for img, cap in batch:
        images.append(img)
        captions.append(cap)
    images = torch.stack(images, dim=0)
    captions = nn.utils.rnn.pad_sequence(
        captions, batch_first=True, padding_value=vocab.word2idx["<PAD>"]
    )
    return images, captions


batch_size = 32

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
)


# Encoder CNN
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.fc(features))
        return features


# Decoder RNN
class DecoderRNN(nn.Module):
    def __init__(
        self, embed_size, hidden_size, vocab_size, embedding_matrix, num_layers=1
    ):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # init hidden state
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)  # [batch_size, seq_len, embed_size]

        # init hidden state with features
        h0 = self.init_h(features).unsqueeze(0)  # [num_layers, batch_size, hidden_size]
        c0 = self.init_c(features).unsqueeze(0)  # [num_layers, batch_size, hidden_size]

        # Passa gli embeddings e lo stato nascosto all'LSTM
        hiddens, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.fc(hiddens)
        return outputs

    def sample(self, features, max_len=20):
        "Genera una didascalia data l'immagine"
        sampled_ids = []

        inputs = features.unsqueeze(1)  # [batch_size, 1, embed_size]
        h0 = self.init_h(features).unsqueeze(0)
        c0 = self.init_c(features).unsqueeze(0)

        states = (h0, c0)
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            if predicted.item() == vocab.word2idx["<END>"]:
                break
        return sampled_ids


def create_embedding_matrix(vocab, glove_embeddings, embedding_dim):
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in vocab.word2idx.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
        else:
            # Inizializza con un vettore casuale per le parole non trovate
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.tensor(embedding_matrix, dtype=torch.float32)


embed_size = embedding_dim
hidden_size = 512
vocab_size = 5900

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, embedding_matrix).to(device)


# Generate caption function
def generate_caption(encoder, decoder, image, vocab, max_length=20):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image = image.to(device)
        feature = encoder(image.unsqueeze(0))
        sampled_ids = decoder.sample(feature, max_length)
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == "<END>":
                break
            sampled_caption.append(word)
        sentence = " ".join(sampled_caption)
    return sentence


# Function to load the model and return caption
def retrieve_model(idx):
    checkpoint = torch.load("ckpt/resnet_rnn.ckpt", map_location=torch.device("cpu"))

    encoder = EncoderCNN(embed_size)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = encoder.to(device)

    decoder = DecoderRNN(embed_size, hidden_size, 5900, embedding_matrix).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    test_image, _ = test_dataset[idx]
    caption = generate_caption(encoder, decoder, test_image, vocab)

    # Generate caption
    return caption
