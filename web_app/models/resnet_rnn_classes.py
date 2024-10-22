import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.models as models
import nltk
nltk.download("punkt_tab")


# Vocabulary class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.word_freq = {}
        self.idx = 4

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

        return image, numericalized_caption, image_id



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
        self, vocab, embed_size, hidden_size, vocab_size, embedding_matrix, num_layers=1
    ):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
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
            if predicted.item() == self.vocab.word2idx["<END>"]:
                break
        return sampled_ids