import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizer
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List 

class SentencesDataset(Dataset):
    """
    Custom dataset of sentences annotated in the format (sentence, sentiment, class)
    """
    def __init__(self, annotations_file: str):

        # Custom data csv
        self.sentences_info = pd.read_csv(annotations_file)

        # Class names and labels
        self.label_map = {
            "positive": 1,
            "negative": 0
        }
        self.class_map = {
            "food": 0,
            "sports": 1,
            "books": 2
        }
        
    def __len__(self):
        return len(self.sentences_info)

    def __getitem__(self, idx: int):
        sentence = self.sentences_info.iloc[idx, 0]

        # Extract class label and one hot encode
        class_label = self.sentences_info.iloc[idx, 1]
        class_label = self.class_map[class_label]
        class_list = np.zeros(3)
        class_list[class_label] = 1

        # Extract sentiment label
        sentiment_label = self.sentences_info.iloc[idx, 2]
        sentiment_label = self.label_map[sentiment_label]

        return sentence, class_list.astype(int), sentiment_label

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Positional encoding following method in paper
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:x.size(0)]
        return x


class SentenceTransformer(nn.Module):

    def __init__(self, tokenizer_name: str='bert-base-cased', d_model: int=512, output_embed_dim: int=368, max_length: int=100):
        super().__init__()

        # Set max length of string
        self.max_length = max_length

        # Tokenizes sentences 
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # Standard Transformer architecture
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fcn = nn.Linear(d_model, output_embed_dim)

    def forward(self, inputs: List[str]):
        # Tokenize inputs before sending to transformer encoder
        tokens = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length)

        # Forward pass
        out = self.embedding(torch.LongTensor(tokens["input_ids"]))
        out = self.positional_encoding(out)
        out = self.encoder(out, src_key_padding_mask=torch.FloatTensor(tokens["attention_mask"]))
        out = self.fcn(out)
        return out

class MTLTransformer(SentenceTransformer):

    def __init__(self, tokenizer_name: str='bert-base-cased', num_classes: int=3, num_sentiments: int=1, d_model: int=512, output_embed_dim: int=368, max_length:int=100):

        super().__init__(tokenizer_name, d_model, output_embed_dim, max_length)

        # Add a fully connected layer for each task
        self.fcn_classification = nn.Linear(output_embed_dim, num_classes)
        self.fcn_sentiment = nn.Linear(output_embed_dim, num_sentiments)

        # Softmax function for outputs
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inputs):
        out = super().forward(inputs)

        # Forward pass for classification
        out_classes = self.fcn_classification(out)
        out_classes = self.softmax(out_classes)

        # Forward pass for sentiment analysis
        out_sentiments = self.fcn_sentiment(out)
        out_sentiments = self.softmax(out_sentiments)
        return out_classes, out_sentiments