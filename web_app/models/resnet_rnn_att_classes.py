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
nltk.download('punkt_tab')

device = torch.device('cpu')
print(device)


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
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
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in tokenized_text]
    


class FlickrDataset(Dataset):
    def __init__(self, dataframe, image_dir, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

        # Creiamo una lista di coppie (immagine, didascalia)
        self.image_ids = []
        self.captions = []

        grouped = dataframe.groupby('image')['caption'].apply(list).reset_index()

        for idx in range(len(grouped)):
            img_id = grouped.loc[idx, 'image']
            captions = grouped.loc[idx, 'caption']
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
    

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: [batch_size, num_pixels, encoder_dim]
        # decoder_hidden: [batch_size, decoder_dim]
        
        att1 = self.encoder_att(encoder_out)      # [batch_size, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden)   # [batch_size, attention_dim]
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # [batch_size, num_pixels]
        alpha = self.softmax(att)                 # weigths attention [batch_size, num_pixels]
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [batch_size, encoder_dim]
        
        return context, alpha


class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Congeliamo i parametri dell'encoder
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        # Estraiamo le feature map
        features = self.resnet(images)  # [batch_size, 2048, feature_map_size, feature_map_size]
        features = features.permute(0, 2, 3, 1)    # [batch_size, feature_map_size, feature_map_size, 2048]
        features = features.view(features.size(0), -1, features.size(-1))  # [batch_size, num_pixels, 2048]
        return features  # Restituisce le feature map spaziali


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab, attention_dim, embed_size, hidden_size, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.vocab = vocab
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)  # attention module

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, hidden_size, bias=True) 
        self.init_h = nn.Linear(encoder_dim, hidden_size) 
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, encoder_out, captions):
        """
        encoder_out: [batch_size, num_pixels, encoder_dim]
        captions: [batch_size, max_caption_length]
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        vocab_size = self.vocab_size

        # Inizializza LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # [batch_size, hidden_size]
        c = self.init_c(mean_encoder_out)

        # Rimuove il token di stop
        embeddings = self.embedding(captions)  # [batch_size, max_caption_length, embed_size]

        # Per salvare le predizioni
        predictions = torch.zeros(batch_size, captions.size(1), vocab_size).to(device)

        for t in range(captions.size(1)):
            batch_size_t = sum([l > t for l in [captions.size(1)]*batch_size])
            context, _ = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # Gate di attenzione
            context = gate * context

            # Concatenazione dell'embedding e del contesto
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], context], dim=1)
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))  # LSTM step

            preds = self.fc(self.dropout(h))  # [batch_size_t, vocab_size]
            predictions[:batch_size_t, t, :] = preds

        return predictions

    def sample(self, encoder_out, max_len=20):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Inizializza LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        # Inizializza la didascalia generata
        sampled_ids = []
        inputs = self.embedding(torch.tensor([self.vocab.word2idx['<START>']]).to(device))  # [1, embed_size]

        for _ in range(max_len):
            context, alpha = self.attention(encoder_out, h)

            gate = self.sigmoid(self.f_beta(h))  # Gate di attenzione
            context = gate * context

            lstm_input = torch.cat([inputs.squeeze(0), context], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))

            preds = self.fc(self.dropout(h))  # [vocab_size]
            predicted = preds.argmax(dim=1)   # [1]
            sampled_ids.append(predicted.item())

            if predicted.item() == self.vocab.word2idx['<END>']:
                break

            inputs = self.embedding(predicted)  # [1, embed_size]

        return sampled_ids