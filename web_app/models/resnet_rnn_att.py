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
import sys
import nltk
from .resnet_rnn_att_classes import Vocabulary, FlickrDataset, EncoderCNN, DecoderWithAttention
sys.modules['__main__'].Vocabulary = Vocabulary
nltk.download("punkt_tab")



class ModelExtractorRnnAtt():

    def __init__(self):
        device = torch.device("cpu")

        device =  torch.device("cpu")

        self.ckpt = torch.load("web_app/ckpt/resnet_rnn_att.ckpt", map_location=torch.device("cpu"))
        self.vocab = self.ckpt["vocab"]
        self.embedding_matrix = self.ckpt["embedding_matrix"]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load the dataset
        captions_train_file = 'web_app/static/images/train/captions_train.txt'
        captions_test_file = 'web_app/static/images/test/captions_test.txt'

        captions_train_df = pd.read_table(captions_train_file, delimiter=',', header=None, names=['image', 'caption'])
        captions_test_df = pd.read_table(captions_test_file, delimiter=',', header=None, names=['image', 'caption'])

        train_image_dir = 'web_app/static/images/train/'
        test_image_dir = 'web_app/static/images/test'

        self.train_dataset = FlickrDataset(captions_train_df, train_image_dir, self.vocab, transform=self.transform)
        self.test_dataset = FlickrDataset(captions_test_df, test_image_dir, self.vocab, transform=self.transform)

        batch_size = 32
        self.train_loader = DataLoader( dataset=self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=self.collate_fn)
        self.test_loader = DataLoader( dataset=self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=self.collate_fn)

        self.attention_dim = 256
        self.embed_size = 100
        self.hidden_size = 512
        self.vocab_size = len(self.vocab)

        self.encoder = EncoderCNN(self.embed_size)
        self.encoder.load_state_dict(self.ckpt["encoder_state_dict"])
        self.encoder = self.encoder.to(device)

        self.decoder = DecoderWithAttention(self.vocab, self.attention_dim, self.embed_size, self.hidden_size, self.vocab_size).to(device)
        self.decoder.load_state_dict(self.ckpt["decoder_state_dict"])
        self.decoder = self.decoder.to(device)


    def collate_fn(self, batch):
        images = []
        captions = []
        for img, cap in batch:
            images.append(img)
            captions.append(cap)
        images = torch.stack(images, dim=0)
        captions = nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=self.vocab.word2idx["<PAD>"])
        return images, captions
    

    def generate_caption(self, idx, max_length=20):
        self.encoder.eval()
        self.decoder.eval()
        device = torch.device("cpu")

        image, _, img_name = self.test_dataset[idx]

        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(device)

            encoder_out = self.encoder(image)  # [batch_size=1, num_pixels, encoder_dim]

            encoder_dim = encoder_out.size(-1)
            encoder_out = encoder_out.view(1, -1, encoder_dim)  # [1, num_pixels, encoder_dim]
            num_pixels = encoder_out.size(1)

            mean_encoder_out = encoder_out.mean(dim=1)  # [1, encoder_dim]
            h = self.decoder.init_h(mean_encoder_out)        # [1, decoder_dim]
            c = self.decoder.init_c(mean_encoder_out)

            sampled_ids = []
            inputs = torch.tensor([self.vocab.word2idx['<START>']]).to(device)
            inputs = self.decoder.embedding(inputs)  # [1, embed_size]

            for _ in range(max_length):
                context, alpha = self.decoder.attention(encoder_out, h)  # context: [1, encoder_dim]

                gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # [1, encoder_dim]
                context = gate * context

                lstm_input = torch.cat([inputs, context], dim=1)  # [1, embed_size + encoder_dim]

                h, c = self.decoder.decode_step(lstm_input, (h, c))  # h, c: [1, hidden_size]

                outputs = self.decoder.fc(self.decoder.dropout(h))  # [1, vocab_size]
                _, predicted = outputs.max(1)             # predicted: [1]

                predicted_id = predicted.item()
                sampled_ids.append(predicted_id)

                if predicted_id == self.vocab.word2idx['<END>']:
                    break

                inputs = self.decoder.embedding(predicted)  # [1, embed_size]

            sampled_caption = [self.vocab.idx2word[word_id] for word_id in sampled_ids]
            sampled_caption.remove('<END>')
            sentence = ' '.join(sampled_caption)

        return sentence, img_name