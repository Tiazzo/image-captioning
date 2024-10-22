import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import nltk
from .resnet_rnn_classes import Vocabulary, FlickrDataset, EncoderCNN, DecoderRNN
sys.modules['__main__'].Vocabulary = Vocabulary
nltk.download("punkt_tab")



class ModelExtractorRnn():
    
    def __init__(self):
        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ckpt = torch.load("web_app/ckpt/resnet_rnn.ckpt", map_location=device)
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

        # Load the model
        self.embed_size = 100
        self.hidden_size = 512
        self.vocab_size = len(self.vocab)

        self.encoder = EncoderCNN(100)
        self.encoder.load_state_dict(self.ckpt["encoder_state_dict"])
        self.encoder = self.encoder.to(device)

        self.decoder = DecoderRNN(self.vocab, self.embed_size, self.hidden_size, self.vocab_size, self.embedding_matrix).to(device)
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


    # Generate caption function
    def generate_caption(self, idx, max_length=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.eval()
        self.decoder.eval()

        image, _, img_name = self.test_dataset[idx]

        with torch.no_grad():
            image = image.to(device)
            feature = self.encoder(image.unsqueeze(0))
            sampled_ids = self.decoder.sample(feature, max_length)
            sampled_caption = []
            for word_id in sampled_ids:
                word = self.vocab.idx2word[word_id]
                if word == "<END>":
                    break
                sampled_caption.append(word)
            sentence = " ".join(sampled_caption)
        
        return sentence, img_name